import pandas as pd
from torch import nn
import torch
from src.demand_prediction.results_functions import wmape
from src.models.EventsModule import EventsModule
from src.models.Discriminator import DiscriminatorModel
import os


class AdversarialEventsModule(EventsModule):
    def __init__(self, hparams, emb_dict, gen_model):
        super().__init__(hparams, emb_dict)
        self.bce_loss = torch.nn.BCELoss()
        self.wd_dis = hparams['weight_decay_dis'] if 'weight_decay_dis' in hparams else 0
        self.lr_dis = hparams['lr_dis'] if 'lr_dis' in hparams else 0.01
        self.g_lmb = hparams['g_lmb'] if 'g_lmb' in hparams else 1
        self.d_lmb = hparams['d_lmb'] if 'd_lmb' in hparams else 1
        self.generator = gen_model
        self.discriminator = DiscriminatorModel(out_dim=self.generator.out_dim, hparams=hparams)
        self.batch_dict, self.lengths, self.x, self.real_days, self.generated_days = None, None, None, None, None
        self.saving_results = hparams['saving_results'] if 'saving_results' in hparams else False
        self.results_prefix = hparams['results_prefix'] if 'results_prefix' in hparams else 'Cosine_HD_'

        self.epoch_logs = {
        "epoch": [],
        "discriminator_fake_acc": [],
        "discriminator_real_acc": [],
        "discriminator_loss": [],
        "generator_loss": [],
        }
        self.log_file_path = f"{self.results_prefix}_epoch_metrics.csv"

        self.test_batch_logs = []
        self.test_log_file = f"{self.results_prefix}_test_batch_metrics.csv"


        self.lstm_feedback_loop = hparams['lstm_feedback_loop'] if 'lstm_feedback_loop' in hparams else False
        if self.lstm_feedback_loop:
            self.epoch_weight_decay_schedule = {
                0: 0.01,
                41: 0.001
            }
            self._prev_cov_gradient_norms = {}
            self._cov_contrib_deltas = []
            self.exact_utility = hparams.get("exact_utility", False)
            self.embedding_scale = None
            self.lstm_gradients = hparams['lstm_gradients'] if 'lstm_gradients' in hparams else True
            self.SEED = 42
            self.lstm_penalty_weight = 0.0
            self.lstm_epochs = hparams['lstm_epochs'] if 'lstm_epochs' in hparams else 1
            self.n_in = hparams['input_chunk_length'] if 'input_chunk_length' in hparams else 10
            self.future_covariates = None
            self.lstm_loss_fn = torch.nn.MSELoss()
            # Load stock data for the downstream task
            self.start_test_date = hparams['start_test_date']

            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.stock_data_path = os.path.join(BASE_DIR, '..', '..', 'data', 'datasets', 'stocks', 'sp500_1980_2018.csv')
            self.stock_data_path = os.path.normpath(self.stock_data_path)
            self.stock_df = pd.read_csv(self.stock_data_path)

            # use tensors for gradient flow
            self.stock_close_values = torch.tensor(
                self.stock_df['Close'].values, 
                dtype=torch.float32, 
                device=torch.device('cpu')
            )
            self._setup_lstm_train_val_test()

            # trying this, no longer recreate it but instead reset the weights
            self.lstm_model = SimpleLSTM(
                target_dim=1,        
                covariate_dim=100,
                hidden_dim=32,
                dropout=0.2
            ).to(self.generator.device)
            self.lstm_optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=1e-3)
            self._pending_lstm_step = False




    def _setup_lstm_train_val_test(self):
        """Prepares raw stock data, splits, and normalizes for LSTM training based on start_test_date."""
        self.stock_df["Date"] = pd.to_datetime(self.stock_df["Date"]).dt.normalize()
        self.stock_df = self.stock_df.sort_values("Date").reset_index(drop=True)
        assert self.stock_df["Date"].is_monotonic_increasing, "Stock data not sorted!"

        # Retrieve `start_test_date` from hyperparameters
        start_test_date = pd.to_datetime(self.start_test_date)

        # Split stock data using the same logic as EventsDataModule
        df_train = self.stock_df[self.stock_df["Date"] < start_test_date].copy()
        df_test = self.stock_df[self.stock_df["Date"] >= start_test_date].copy()

        if df_test.empty:
            raise ValueError(f"No test data found after {start_test_date}. Check stock data range.")

        # Normalize stock prices
        y_train = torch.tensor(df_train["Close"].values, dtype=torch.float32)
        y_test  = torch.tensor(df_test["Close"].values, dtype=torch.float32)
        mean, std = y_train.mean(), y_train.std()
        y_train = (y_train - mean) / std
        y_test = (y_test - mean) / std

        self.train_target = y_train.unsqueeze(-1)  # shape: (T, 1)
        self.test_target = y_test.unsqueeze(-1)    # shape: (T, 1)

        # Extract train/test dates
        self.train_dates = df_train["Date"].tolist()
        self.test_dates = df_test["Date"].tolist()

        self.df_train = df_train
        self.df_test = df_test

        print(f"Stock Train/Test Split Aligned with EventsDataModule:")
        print(f"   - Train: {len(self.train_dates)} days ({self.train_dates[0]} --> {self.train_dates[-1]})")
        print(f"   - Test:  {len(self.test_dates)} days ({self.test_dates[0]} --> {self.test_dates[-1]})")



    @staticmethod
    def adversarial_accuracy(y_pred, y_true):
        values = (y_pred >= 0.5).squeeze(dim=1).int()
        true_values = y_true.squeeze(dim=1)
        assert values.size(0) > 0 and values.size(0) == true_values.size(0)
        return torch.eq(values, true_values).sum().item() / values.size(0)

    def adversarial_loss(self, y_hat, y):
        return self.bce_loss(y_hat, y)

    def forward(self, x, test_mode=False):
        return self.generator(x, test_mode)

    def step(self, batch: dict, optimizer_idx: int, name: str = 'loss', batch_idx: int = None):
        results, loss, dis_loss, gen_loss = {}, None, None, 0

        # train generator
        if optimizer_idx == 0:
            self.batch_dict, self.lengths = batch
            self.x = self.generator.embedding_model(self.batch_dict)
            self.real_days = self.x.detach().clone()
            if self.embedding_scale is None:
                # take the RMS norm across batch and events as our “scale”
                # real_days: [B, T, D] --> flatten to [B*T, D]
                flat = self.real_days.reshape(-1, self.real_days.size(-1))
                # RMS norm: sqrt(mean(||x||²))
                self.embedding_scale = flat.norm(dim=-1).square().mean().sqrt().item()
            self.generated_days, mask_indices = self((self.x, self.lengths))  # generate counterfactual days

            for row_idx, row_len in enumerate(self.lengths):
                masked_predict = self.generated_days[row_idx][mask_indices[row_idx]]
                masked_target = self.real_days[row_idx][mask_indices[row_idx]]
                gen_loss += self.gen_loss(masked_predict, masked_target)
            gen_loss /= self.x.size(0)

            ###############
            if self.lstm_feedback_loop and name == 'train' and self.current_epoch >= 1:
                self.lstm_penalty_weight = min(10.0, 0.2 + 0.25 * self.current_epoch)
                lstm_batch_wmape = self.downstream_task(batch, batch_idx)
                results[f'generator/{name}_lstm_batch_wmape'] = lstm_batch_wmape.item()
                gen_loss = gen_loss + (self.lstm_penalty_weight * lstm_batch_wmape)       # commented out to test effect
            ###############

            # restore the unmasked events only
            self.generated_days[~mask_indices] = self.real_days[~mask_indices]

            # Here the optimizer is not the discriminator's optimizer so it's legal to call the discriminator forward
            y_pred = self.discriminator((self.generated_days, self.lengths))    # make predictions
            y_true = torch.ones_like(y_pred)    # ground truth - all fake
            dis_loss = self.adversarial_loss(y_pred, y_true)    # adversarial loss is binary cross-entropy
            acc = AdversarialEventsModule.adversarial_accuracy(y_pred.detach(), y_true.detach())

            loss = self.g_lmb * gen_loss + self.d_lmb * dis_loss
            wmape_val = results.get("generator/train_lstm_batch_wmape", None)
            if wmape_val is not None:
                wmape_val = wmape_val.item() if isinstance(wmape_val, torch.Tensor) else wmape_val
                self.logged_batches["wmape"].append(wmape_val)
            results[f'generator/{name}_loss'] = loss
            results[f'generator/{name}_discriminator_accuracy'] = acc
            results[f'generator/{name}_generator_loss'] = gen_loss
            results[f'generator/{name}_discriminator_loss'] = dis_loss

        # train discriminator & measuring discriminator's ability to classify real from generated samples
        if optimizer_idx == 1:
            # the following logic weakens the discriminator with scheduled real input noise
            if self.lstm_feedback_loop:
                  
                max_eps=10-self.lstm_penalty_weight # was 0.005
                min_eps = 0.0001               
                T=100
                progress = min(1.0, self.current_epoch / T)
                eps_unit = max_eps * (1-progress) + min_eps * progress
                epsilon  = eps_unit * self.embedding_scale # in units of embedding_scale

                def cosine_perturb(x, epsilon=0.1):
                    norm_x = x.norm(dim=-1, keepdim=True) + 1e-8
                    unit_x = x / norm_x
                    noise = torch.randn_like(x)
                    noise = noise - (noise * unit_x).sum(dim=-1, keepdim=True) * unit_x
                    noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-8)
                    perturbed = unit_x + epsilon * noise
                    return perturbed * norm_x  # restore original norm

                real_input = cosine_perturb(self.real_days.detach(), epsilon=epsilon)
                fake_input = cosine_perturb(self.generated_days.detach(), epsilon=epsilon)

                y_pred_real = self.discriminator((real_input, self.lengths))
                min_real = 0.99
                max_real = min(1.0, 0.99 + self.lstm_penalty_weight * 0.1)
                y_real = torch.empty_like(y_pred_real).uniform_(min_real, max_real)

                real_loss = self.adversarial_loss(y_pred_real, y_real)
                real_acc = AdversarialEventsModule.adversarial_accuracy(y_pred_real.detach(), (y_real >= 0.5).int())

                y_pred_fake = self.discriminator((fake_input, self.lengths))
                y_fake = torch.zeros_like(y_pred_fake)
                fake_loss = self.adversarial_loss(y_pred_fake, y_fake)
                fake_acc = AdversarialEventsModule.adversarial_accuracy(y_pred_fake.detach(), y_fake.int())

                loss = (real_loss + fake_loss) / 2

            else:
                # === B: Simpler Discriminator Logic ===
                y_pred = self.discriminator((self.real_days, self.lengths))
                y_true = torch.ones_like(y_pred)
                real_loss = self.adversarial_loss(y_pred, y_true)
                real_acc = AdversarialEventsModule.adversarial_accuracy(y_pred.detach(), y_true.detach())

                y_pred_fake = self.discriminator((self.generated_days.detach(), self.lengths))
                y_fake = torch.zeros_like(y_pred_fake)
                fake_loss = self.adversarial_loss(y_pred_fake, y_fake)
                fake_acc = AdversarialEventsModule.adversarial_accuracy(y_pred_fake.detach(), y_fake.detach())

                loss = (real_loss + fake_loss) / 2

            results[f'discriminator/{name}_discriminator_real_loss'] = real_loss
            results[f'discriminator/{name}_discriminator_fake_loss'] = fake_loss
            results[f'discriminator/{name}_discriminator_loss'] = loss
            results[f'discriminator/{name}_discriminator_fake_acc'] = fake_acc
            results[f'discriminator/{name}_discriminator_real_acc'] = real_acc

        for k, v in results.items():
            self.log(k, v)

        return loss, results

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        loss, results = self.step(batch, optimizer_idx, name='train', batch_idx=batch_idx)

        # experimenting with logging
        if self.current_epoch == 0:
            self.logged_batches = {
                "fake_acc": [],
                "real_acc": [],
                "dis_loss": [],
                "gen_loss": [],
                "wmape": [],
                "cov_contrib": [],
            }

        if optimizer_idx == 1:
            # discriminator step
            fake_acc = results.get("discriminator/train_discriminator_fake_acc", 0)
            real_acc = results.get("discriminator/train_discriminator_real_acc", 0)
            dis_loss = results.get("discriminator/train_discriminator_loss", 0)

            # unwrap any tensors
            fake_acc = fake_acc.item() if isinstance(fake_acc, torch.Tensor) else fake_acc
            real_acc = real_acc.item() if isinstance(real_acc, torch.Tensor) else real_acc
            dis_loss = dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss

            self.logged_batches["fake_acc"].append(fake_acc)
            self.logged_batches["real_acc"].append(real_acc)
            self.logged_batches["dis_loss"].append(dis_loss)

        if optimizer_idx == 0:
            # generator step
            gen_loss = results.get("generator/train_generator_loss", 0)
            gen_loss = gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss
            self.logged_batches["gen_loss"].append(gen_loss)
            cov_contrib = results.get("generator/train_cov_contrib", None)
            if cov_contrib is not None:
                cov_contrib = cov_contrib.item() if isinstance(cov_contrib, torch.Tensor) else cov_contrib
                self.logged_batches["cov_contrib"].append(cov_contrib)

        if optimizer_idx == 1 and getattr(self, '_pending_lstm_step', False):
            self.lstm_optimizer.step()
            self.lstm_optimizer.zero_grad()
            self._pending_lstm_step = False

        return {'loss': loss, 'prog': results}

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        results = {}
        for i in range(len(self.optimizers())):
            results.update(self.step(batch, i, name='val', batch_idx=batch_idx)[1])
        return results

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        results = {}
        for i in range(len(self.optimizers())):
            results.update(self.step(batch, i, name='test', batch_idx=batch_idx)[1])
        
        # Save this batch's results to list
        row = {'batch_idx': batch_idx}
        row.update({k: float(v) for k, v in results.items() if isinstance(v, (int, float, torch.Tensor))})
        self.test_batch_logs.append(row)

        return results

    def get_embeddings_generator(self, data_loader, dates):
        embeddings = None
        for batch in data_loader:
            batch_dict, lengths = batch
            x = self.generator.embedding_model(batch_dict).to(self.device)
            out, weights_matrix_list = self.generator((x, lengths), test_mode=True)
            new_out = torch.zeros((out.shape[0], out.shape[2]), dtype=torch.float, device=self.device)
            for row_idx, row_len in enumerate(lengths):
                masked_event = self.get_masked_embedding()
                for event_idx in range(row_len):
                    cur_x = batch_dict['embeddings'][row_idx][:row_len].float().to(self.device)
                    if row_len > 1:
                        cur_x[event_idx] = masked_event
                    cur_out, _ = self.generator((cur_x.unsqueeze(0), torch.tensor([row_len])), test_mode=True)
                    cur_out = cur_out[0][event_idx]
                    new_out[row_idx] += cur_out
                new_out[row_idx] /= row_len
            embeddings = new_out if embeddings is None else torch.cat([embeddings, new_out])
        df = pd.DataFrame(list(zip(dates[:len(embeddings)], embeddings.tolist())), columns=['date', 'embeddings'])
        return df

    def on_test_end(self):
        print("Test end function:")
        if self.saving_results:
            print("Start testing GAN")
            embedding_train_dates, train_dates, val_dates, test_dates = self.trainer.datamodule.get_dates()
            results_prefix = self.results_prefix
            df_train = self.get_embeddings_generator(self.trainer.datamodule.embedding_train_dataloader(), embedding_train_dates)
            df_test = self.get_embeddings_generator(self.trainer.datamodule.test_dataloader(), test_dates)
            df_train.to_pickle("gan_embeddings/" + results_prefix + "_train_gan_embeddings.pkl")
            df_test.to_pickle("gan_embeddings/" + results_prefix + "_test_gan_embeddings.pkl")

        # Save batch test results to csv
        if self.test_batch_logs:
            df = pd.DataFrame(self.test_batch_logs)
            avg_fake_acc = df["discriminator/test_discriminator_fake_acc"].mean()
            avg_real_acc = df["discriminator/test_discriminator_real_acc"].mean()
            print(f"--> Avg Fake Accuracy: {avg_fake_acc:.3f}")
            print(f"--> Avg Real Accuracy: {avg_real_acc:.3f}")
            df.to_csv(self.test_log_file, index=False)
            print(f"[Test Summary] Wrote per-batch results to: {self.test_log_file}")

    def configure_optimizers(self):
        optimizer1 = torch.optim.AdamW(self.generator.parameters(), lr=self.lr_gen, weight_decay=self.wd_gen)
        optimizer2 = torch.optim.AdamW(self.discriminator.parameters(), lr=self.lr_dis, weight_decay=self.wd_dis)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.95, last_epoch=-1)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.95, last_epoch=-1)
        return [optimizer1, optimizer2], [scheduler1, scheduler2]


    def on_train_epoch_start(self, outputs=None):
        if self.current_epoch in self.epoch_weight_decay_schedule:
            new_wd = self.epoch_weight_decay_schedule[self.current_epoch]
            if self.wd_dis != new_wd:  # only update if changed
                print(f"--> Updating discriminator weight decay to {new_wd}")
                self.wd_dis = new_wd
                for g in self.optimizers()[1].param_groups:
                    g['weight_decay'] = new_wd



    def on_train_epoch_end(self, outputs=None):
        # If this is the end of Epoch zero, we want to create and fit
        # nn.lstm with the first set of generated_days.
        # we call downstream_task() from here to ensure correct timing, but every other call
        # to downstream_task() will be made in the training_step per batch.
        if self.current_epoch == 1 and self.lstm_feedback_loop:
            x = self.downstream_task() # we won't use these yet -- this call simply began

        epoch = self.current_epoch
        logs = self.logged_batches

        self.epoch_logs["epoch"].append(epoch)
        self.epoch_logs["discriminator_fake_acc"].append(sum(logs["fake_acc"]) / len(logs["fake_acc"]) if logs["fake_acc"] else 0)
        self.epoch_logs["discriminator_real_acc"].append(sum(logs["real_acc"]) / len(logs["real_acc"]) if logs["real_acc"] else 0)
        self.epoch_logs["discriminator_loss"].append(sum(logs["dis_loss"]) / len(logs["dis_loss"]) if logs["dis_loss"] else 0)
        self.epoch_logs["generator_loss"].append(sum(logs["gen_loss"]) / len(logs["gen_loss"]) if logs["gen_loss"] else 0)
        self.epoch_logs["generator_avg_wmape"] = self.epoch_logs.get("generator_avg_wmape", [])
        self.epoch_logs["generator_avg_wmape"].append(sum(logs["wmape"]) / len(logs["wmape"]) if logs["wmape"] else 0)
        self.epoch_logs["generator_cov_contrib"] = self.epoch_logs.get("generator_cov_contrib", [])
        self.epoch_logs["generator_cov_contrib"].append(sum(logs["cov_contrib"]) / len(logs["cov_contrib"]) if logs["cov_contrib"] else 0)
        
        deltas_only = [d[2] for d in self._cov_contrib_deltas]  # just the deltas
        avg_delta = sum(deltas_only) / len(deltas_only) if deltas_only else 0
        self.epoch_logs.setdefault("generator_avg_cov_gradient_delta", []).append(avg_delta)

        if self.saving_results:
            delta_df = pd.DataFrame(self._cov_contrib_deltas, columns=["epoch", "batch_idx", "delta"])
            delta_file = f"{self.results_prefix}_gradient_deltas.csv"
            # delta_df.to_csv(delta_file, mode='a', header=not os.path.exists(delta_file), index=False)

        
        self._cov_contrib_deltas = [] # reset

        df = pd.DataFrame(self.epoch_logs)
        df.to_csv(self.log_file_path, index=False)

        print(f"[Epoch {epoch}] Saved logs to {self.log_file_path}")
            
    def reset_weights(self, model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def downstream_task(self, batch=None, batch_idx=None):
        """
        Handles two scenarios:
        1) At epoch 1: trains LSTM once on full dataset (with all covariates).
        2) Afterward: uses batch-only slices of covariates + targets.
        """

        if self.current_epoch == 1:
            train_dates = self.train_dates
            emb_dim = self.generated_days[0][0].shape[0]

            y = self.train_target.to(self.generator.device)
            if y.ndim == 1:
                y = y.unsqueeze(-1)  # (T, 1)

            batch_dates = [pd.to_datetime(d).normalize() for d in self.batch_dict.get("date", [])]
            date_to_idx = {dt: i for i, dt in enumerate(batch_dates)}

            future_cov_list = []
            for dt in train_dates:
                if dt in date_to_idx:
                    batch_idx = date_to_idx[dt]
                    new_emb = self.generated_days[batch_idx, 0].detach().clone()
                else:
                    new_emb = torch.zeros(emb_dim, device=self.generated_days.device)
                future_cov_list.append(new_emb)

            future_cov_tensor = torch.stack(future_cov_list, dim=0)  # (T, emb_dim)
            self.future_covariates_tensor = future_cov_tensor.detach()  # Save without grad

            self.lstm_model.train()
            self.lstm_optimizer.zero_grad()

            min_len = min(len(y), len(future_cov_tensor))
            y = y[-min_len:]
            cov_full = future_cov_tensor[-min_len:]

            h = 1 # horizon-1 shift
            if y.size(0) <= h:
                return torch.tensor(0.0, device=self.generator.device, requires_grad=True)

            y_in   = y[:-h].unsqueeze(0)          # (1, T-h, 1)
            cov_in = cov_full[:-h].unsqueeze(0)   # (1, T-h, D)
            y_true = y[h:].unsqueeze(0)           # (1, T-h, 1)
            # ────────────────────────────────────────────────────────────────

            self.lstm_model.train()
            self.lstm_optimizer.zero_grad()

            output = self.lstm_model(y_in, cov_in)   # shifted tensors
            loss = self.wmape(y_true, output)

            return loss

        elif self.current_epoch >= 1:
            batch_dict, _ = batch
            train_dates = self.train_dates

            batch_dates = [pd.to_datetime(d).normalize() for d in batch_dict["date"]]
            train_dates_normalized = [pd.to_datetime(d).normalize() for d in train_dates]

            batch_date_to_idx = {dt: i for i, dt in enumerate(batch_dates)}
            train_date_to_idx = {dt: i for i, dt in enumerate(train_dates_normalized)}

            y = self.train_target.to(self.generator.device)
            if y.ndim == 1:
                y = y.unsqueeze(-1)

            y_batch_list = []
            cov_batch_list = []

            for dt in batch_dates:
                if dt in train_date_to_idx and dt in batch_date_to_idx:
                    train_idx = train_date_to_idx[dt]
                    batch_idx = batch_date_to_idx[dt]

                    y_batch_list.append(y[train_idx])
                    emb = self.generated_days[batch_idx, :self.lengths[batch_idx]].mean(dim=0)
                    cov_batch_list.append(emb.clone() if self.lstm_gradients else emb.clone().detach())
                else:
                    print(f"Warning!! date {dt} not found in train or batch.")

            if len(cov_batch_list) == 0:
                raise ValueError("No valid dates found inbatch.")

            y_batch = torch.stack(y_batch_list, dim=0)           # (T, 1)
            cov_batch = torch.stack(cov_batch_list, dim=0)       # (T, emb_dim)

            
            h = 1 
            if y_batch.size(0) <= h:            # bounds check
                return torch.tensor(0.0, device=self.generator.device, requires_grad=True)

            y_in = y_batch[:-h].unsqueeze(0)  # (1, T-h, 1)
            cov_in = cov_batch[:-h].unsqueeze(0)# (1, T-h, D)
            y_true = y_batch[h:].unsqueeze(0)   # (1, T-h, 1)
            cov_in.retain_grad()
            # ────────────────────────────────────────────────────────────────

            self.lstm_model.train()

            for _ in range(self.lstm_epochs):
                self.lstm_optimizer.zero_grad()
                output = self.lstm_model(y_in, cov_in)   
                loss = self.wmape(y_true, output)
                loss.backward(retain_graph=True)         # keep grads for generator

                # gradient-norm bookkeeping for ablation tests
                if self.lstm_gradients and batch_idx is not None:
                    grad = cov_in.grad.detach() if cov_in.grad is not None else None
                    if grad is not None:
                        gnorm = grad.norm(dim=-1).mean().item()
                        self.logged_batches["cov_contrib"].append(gnorm)
                        prev = self._prev_cov_gradient_norms.get(batch_idx)
                        if prev is not None:
                            delta = abs(gnorm - prev)
                            self._cov_contrib_deltas.append(
                                (self.current_epoch, batch_idx, delta)
                            )
                        self._prev_cov_gradient_norms[batch_idx] = gnorm

            self._pending_lstm_step = True
            return loss
        
    def wmape(self, y_true, y_pred):
        """
        Compute Weighted Mean Absolute Percentage Error (WMAPE)
        WMAPE = sum(|y - ŷ|) / sum(|y|)
        """
        numerator = torch.sum(torch.abs(y_true - y_pred))
        denominator = torch.sum(torch.abs(y_true)) + 1e-8  # add small val so we never divide by zero
        return numerator / denominator

        
        

class SimpleLSTM(nn.Module):
    def __init__(self, target_dim, covariate_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.input_dim = target_dim + covariate_dim
        self.rnn = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, target_seq, future_covariates):
        """
        Inputs:
            target_seq:        shape (B, T, target_dim)
            future_covariates: shape (B, T, covariate_dim)
        Output:
            prediction:        shape (B, T, 1)
        """
        x = torch.cat([target_seq, future_covariates], dim=-1)  # shape: (B, T, target_dim + cov_dim)
        out, _ = self.rnn(x)
        out = self.dropout(out)
        out = self.fc(out)  # shape: (B, T, 1)
        return out

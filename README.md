# An Articulation of Natural Language for Forecasting: Next Steps in the Evolution of Relevance

## Kenneth Luke Skertich


> **Abstract:**

	Forecasting models often become unreliable during volatile periods caused by anomalous events, from COVID-19, earthquakes, or even sports championships. Recent efforts have shown that supporting these models with external information about the world can significantly boost performance. However, there is no consensus in literature about what information in the world is relevant for the forecasting task. Without clear definitions, the current literature fails to address the core natural language representation trade-off between noise and language richness. 
We formalize the role of natural language in forecasting by introducing a new framework for defining relevance and utility. Then, to operationalize new definitions, we introduce Task-Aware-Covariates (TAC) generated from a Generative Adversarial Network (GAN), a novel utility-optimizable natural language embedding framework for forecasting. We define the most relevant information as that which improves the downstream task. 
TAC-GAN uses a transformer-based GAN to encode Wikipedia page embeddings, guided by two objectives: learning the relationships between the events of a day through a reconstruction task, and minimizing forecasting error on the quantity prediction. 
We introduce an original mathematical underpinning of time series volatility to bridge theoretical gaps in multi-data stream forecasting. Our approach reframes forecasting under volatility as a learnable optimization of relevance itself, offering a principled foundation for a new class of models. 
We apply TAC-GAN’s learned embeddings as covariates to an LSTM model forecasting the S&P 500 index for proof-of-concept, demonstrating improved performance over GAN-Event LSTM under anomalous conditions.
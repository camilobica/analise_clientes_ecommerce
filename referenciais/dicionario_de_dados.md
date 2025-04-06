# Dicionário de Dados – Análise de Clientes (E-commerce)

## Variáveis Categóricas

- `Gender` *(object)* – Gênero do cliente (Male, Female, Prefer not to say).
- `Item Purchased` *(object)* – Nome do item adquirido pelo cliente.
- `Category` *(object)* – Categoria geral do item adquirido (ex: Apparel, Footwear, Accessories).
- `Location` *(object)* – Localização do cliente no momento da compra.
- `Size` *(object)* – Tamanho do item adquirido (S, M, L, XL, etc.).
- `Color` *(object)* – Cor principal do item adquirido.
- `Season` *(object)* – Estação do ano em que a compra foi realizada (Winter, Summer, etc.).
- `Shipping Type` *(object)* – Tipo de envio selecionado (Standard, Express, etc.).
- `Payment Method` *(object)* – Forma de pagamento utilizada (Credit Card, PayPal, etc.).
- `Frequency of Purchases` *(object)* – Frequência com que o cliente costuma comprar (Rarely, Occasionally, Frequently).
- `Age Interval` *(object)* – Faixa etária do cliente (ex: 18–25, 26–35, etc.), criada para facilitar segmentações.
- `Review Interval` *(object)* – Intervalo categorizado da nota de avaliação (ex: Baixa, Média, Alta).

## Variáveis Numéricas

- `Age` *(int64)* – Idade do cliente no momento da compra.
- `Purchase Amount (USD)` *(int64)* – Valor total gasto na compra (em dólares).
- `Review Rating` *(float64)* – Nota de avaliação do produto dada pelo cliente (escala de 1 a 5).
- `Subscription Status` *(int64)* – Indica se o cliente é assinante da loja (0 = não, 1 = sim).
- `Discount Applied` *(int64)* – Indica se a compra teve desconto aplicado (0 = não, 1 = sim).
- `Previous Purchases` *(int64)* – Número de compras anteriores realizadas pelo cliente.

# Dataset Overview
## example of dataset
```
,customer_id,geohash,order_id,vendor_id,product_id,day_of_week,order_time,order_day
0,90a4e98622,u6sc4,0,e1f3e4a4,9971ae2cd1ba,3,16:00:16,11 days
1,90a4e98622,u6sc4,0,e1f3e4a4,00734c4b351f,3,16:00:16,11 days
2,90a4e98622,u6sc4,1,5d1b1300,9a2b00f39640,1,16:34:04,51 days
3,90a4e98622,u6sc4,2,5d1b1300,9a2b00f39640,6,14:39:12,84 days
4,90a4e98622,u6sc4,3,5d1b1300,9a2b00f39640,3,16:23:14,53 days
5,90a4e98622,u6sc4,4,4790e97d,f6b685cff997,2,15:34:13,66 days
6,90a4e98622,u6sc4,5,2ebe1498,02bffeb55fa2,6,16:38:57,49 days
```

## example of product

```
,vendor_id,product_id,name,unit_price
0,e21306e2,055e72373069,Small,0.496
1,7b111c3f,ec33db14a2c5,Alcachofas Empinzadas,0.304
2,4dc46e0f,fa04d9e5b964,Coca-Cola 2L,0.24
3,14d2b0c7,9f502de9373e,"5 Lax, 5 Räkor, 5 Avokado",0.82
```

## example of vendors
```
,vendor_id,chain_id,geohash,primary_cuisine
0,a5028ec5,,u6sc9,thai
1,fed855cd,,u6sc7,thai
2,5f4ad3ab,,u6sc8,sushi
3,83b6b60e,,u6sce,asiatiskt
4,8978d654,,u6scd,asiatiskt
```

## data schema
| Column           | Description                                                                 | Dataset                   |
|------------------|------------------------------------------------------------------------------|---------------------------|
| customer_id      | uniquely hashed customer_id                                                  | Orders                    |
| order_time       | time order was placed                                                        | Orders                    |
| day_of_week      | day of the week the order was placed, with 0 = Sunday                        | Orders                    |
| order_day        | day from the start of the dataset when the order was placed                  | Orders                    |
| geohash          | hashed geographic location of where the order was placed given to 5 digits of precision | Orders, Vendors           |
| order_id         | unique order_id                                                              | Orders                    |
| vendor_id        | unique hashed vendor_id of where the order was placed                        | Orders, Vendors, Product  |
| chain_id         | unique hashed identifier of the chain that a vendor belongs to. Not all vendors belong to a chain and if null the vendor is not part of a chain | Vendor |
| geohash          | hashed geographic location of where the order was placed given to 5 digits of precision | Orders, Vendor            |
| primary_cuisine  | vendor specified cuisine                                                     | Vendor                    |
| product_id       | unique hashed product_id of the product ordered                              | Orders, Product           |
| name             | the product name as displayed on the menu                                    | Product                   |
| unit_price       | the normalised unit price of the item                                        | Product                   |


Great question — this paper is actually very clear once you map **their abstract description** to **your concrete order log**.

# **step-by-step how the paper prepares a real dataset like yours for sequential recommendation**

## 1️⃣ What is a “sequence” in this paper?

The paper frames **sequential recommendation** as:

> Given a sequence of user interactions
> ( S_u = (s_1, s_2, \dots, s_n) ),
> predict the next item ( s_{n+1} ). 

### In your dataset

Each row is **one item interaction**, not a sequence yet:

```
customer_id, order_id, product_id, order_time, order_day, ...
```

So the **first transformation** is:

### ➜ Group rows into sequences per user, ordered by time

For your example:

```text
customer_id = 90a4e98622

(order_day, order_time, product_id)
----------------------------------
11 days, 16:00:16, 9971ae2cd1ba
11 days, 16:00:16, 00734c4b351f
51 days, 16:34:04, 9a2b00f39640
84 days, 14:39:12, 9a2b00f39640
53 days, 16:23:14, 9a2b00f39640
66 days, 15:34:13, f6b685cff997
49 days, 16:38:57, 02bffeb55fa2
```

After sorting by `(order_day, order_time)` you get a **chronological item sequence**:

```text
S_u = [
  product_A,
  product_B,
  product_C,
  ...
]
```

This is the **core sequential signal** used everywhere in the paper.

---

## 2️⃣ How do they turn sequences into training samples?

This is **crucial**.

The paper uses the **standard next-item prediction setup**:

> The session (or sequence) contains all items except the last one,
> and the last item is the ground truth target. 

### Sliding-window construction

From a sequence:

```
[A, B, C, D, E]
```

They generate multiple training samples:

| Input sequence | Target |
| -------------- | ------ |
| [A]            | B      |
| [A, B]         | C      |
| [A, B, C]      | D      |
| [A, B, C, D]   | E      |

This applies to:

* **GRU4Rec**
* **SASRec**
* **BERT4Rec**
* **LLMSeqPrompt**
* **LLMSeqSim**
* **LLM2Sequential**

---

## 3️⃣ Which columns do they actually use?

Despite your dataset having many columns, **most are ignored** for modeling.

### ✅ Always used

| Column                    | Usage                                     |
| ------------------------- | ----------------------------------------- |
| `customer_id`             | grouping sequences                        |
| `product_id`              | item identity (the thing being predicted) |
| `order_day`, `order_time` | sorting only                              |

### ❌ NOT used directly in models

* `geohash`
* `vendor_id`
* `day_of_week`
* `order_id`

The paper explicitly focuses on **item sequences**, not context features, for the sequential models .

---

## 4️⃣ How items become “tokens” (ID-based models)

For **GRU4Rec / SASRec / BERT4Rec**:

* Each `product_id` → mapped to an integer index
* Embedding lookup:

  ```
  product_id → embedding vector
  ```

This is identical to NLP tokenization, but with **items instead of words**.

---

## 5️⃣ What changes when LLMs are involved?

The paper introduces **three ways** LLMs change dataset preparation.

---

### 🟦 A) LLMSeqSim (embedding-only, no training)

**Data preparation:**

1. Use **product metadata** (usually product name)
2. Call LLM embedding API once per product
3. Store embedding vector per `product_id`
4. For a session:

   * Look up embeddings of all items
   * Aggregate them into a **session embedding**
   * Recommend nearest items by cosine similarity 

**Important:**
Your dataset’s `product_id` → joins with a **Product table** to get product names.

---

### 🟦 B) LLMSeqPrompt (fine-tuning LLMs)

This is where dataset formatting becomes **textual**.

#### Training example construction

From a sequence:

```
[A, B, C, D]
```

They create a **prompt-completion pair**:

```json
{
  "prompt": "A, B, C",
  "completion": "D"
}
```

(Variants include ranking, classification, or list generation.)

This is explicitly described in Section 4 of the paper .

**Key point:**
The sequence is converted into **natural language item names**, not IDs.

---

### 🟦 C) LLM2Sequential (your most relevant case)

This is likely what you care about most.

**Preparation pipeline:**

1. Use **product names** → get LLM embeddings
2. Optionally reduce dimension (PCA / Autoencoder)
3. Initialize the **embedding layer** of:

   * BERT4Rec
   * SASRec
   * GRU4Rec
4. Train normally using item sequences 

> “We supply item embeddings retrieved from an LLM as input to the embedding layer” 

So:

* Your dataset stays **ID-based**
* Only the **initial embedding values** come from LLMs

---

## 7️⃣ One subtle but important detail

The paper **does NOT**:

* Use order quantities
* Use vendor transitions
* Use geospatial modeling
* Use time gaps as features

Time is **only used to order events**, not as an input feature .
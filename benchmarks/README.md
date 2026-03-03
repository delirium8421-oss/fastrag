## Benchmarks
We validate the benchmark results provided in [HippoRAG](https://arxiv.org/abs/2405.14831), as well as comparing with other methods:
- NaiveRAG (vector dbs) using the OpenAI embedder `text-embedding-3-small`
- [LightRAG](https://github.com/HKUDS/LightRAG) 
- [GraphRAG](https://github.com/gusye1234/nano-graphrag) (we use the implementation provided by `nano-graphrag`, based on the original [Microsoft GraphRAG](https://github.com/microsoft/graphrag))

### Results
**2wikimultihopQA**
| # Queries |  Method  | All queries % | Multihop only % |
|----------:|:--------:|--------------:|----------------:|
|         51||||
|           |  VectorDB|           0.49|             0.32|
|           |  LightRAG|           0.47|             0.32|
|           |  GraphRAG|           0.75|             0.68|
|           |**Circlemind**|           **0.96**|             **0.95**|
|        101||||
|           |  VectorDB|           0.42|             0.23|
|           |  LightRAG|           0.45|             0.28|
|           |  GraphRAG|           0.73|             0.64|
|           |**Circlemind**|           **0.93**|             **0.90**|

**Circlemind is up to 4x more accurate than VectorDB RAG.**

**HotpotQA**
| # Queries |  Method  | All queries % |
|----------:|:--------:|--------------:|
|        101|||
|           |  VectorDB|           0.78|
|           |  LightRAG|           0.55|
|           |  GraphRAG|             -*|
|           |**Circlemind**|           **0.84**|

*: crashes after half an hour of processing

Below, find the insertion times for the 2wikimultihopqa benchmark (~800 chunks):
|  Method  |  Time (minutes)  |
|:--------:|-----------------:|
|  VectorDB|              ~0.3|
|  LightRAG|               ~25|
|  GraphRAG|               ~40|
|**Circlemind**|              ~1.5|

**Circlemind is 27x faster than GraphRAG while also being over 40% more accurate in retrieval.**

### Run it yourself
The scripts in this directory will generate and evaluate the 2wikimultihopqa datasets on a subsets of 51 and 101 queries with the same methodology as in the HippoRAG paper. In particular, we evaluate the retrieval capabilities of each method, mesauring the percentage of queries for which all the required evidence was retrieved. We preloaded the results so it is enough to run `evaluate_dbs.xx` to get the numbers. You can also run `create_dbs.xx` to regenerate the databases for the different methods.  

A couple of NOTES:
- you will need to set an OPENAI_API_KEY;
- LightRAG and GraphRAG could take a over an 1 hour to process and they can be expensive;
- when pip installing LightRAG, not all dependencies are added; to run it we simply deleted all the imports of each missing dependency (since we use OpenAI they are not necessary).
- we also benchmarked on the HotpotQA dataset (we will soon release the code for that as well).

The output will look similar to the following (the exact numbers could vary based on your graph configuration)
```
Evaluation of the performance of different RAG methods on 2wikimultihopqa (51 queries)

VectorDB
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.49019607843137253
[multihop only] Percentage of queries with perfect retrieval: 0.32432432432432434

LightRAG [local mode]
Loading dataset...
Percentage of queries with perfect retrieval: 0.47058823529411764
[multihop only] Percentage of queries with perfect retrieval: 0.32432432432432434

GraphRAG [local mode]
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.7450980392156863
[multihop only] Percentage of queries with perfect retrieval: 0.6756756756756757

Circlemind
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.9607843137254902
[multihop only] Percentage of queries with perfect retrieval: 0.9459459459459459


Evaluation of the performance of different RAG methods on 2wikimultihopqa (101 queries)

VectorDB
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.4158415841584158
[multihop only] Percentage of queries with perfect retrieval: 0.2318840579710145

LightRAG [local mode]
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.44554455445544555
[multihop only] Percentage of queries with perfect retrieval: 0.2753623188405797

GraphRAG [local mode]
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.7326732673267327
[multihop only] Percentage of queries with perfect retrieval: 0.6376811594202898

Circlemind
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.9306930693069307
[multihop only] Percentage of queries with perfect retrieval: 0.8985507246376812
```

echo "Evaluation of the performance of different RAG methods on the 2wikimultihopqa (51 queries)";
echo;
echo "VectorDB";
python vdb_benchmark.py -n 51 -s
echo;
echo "LightRAG [local mode]";
python lightrag_benchmark.py -n 51 -s --mode=local
echo;
echo "GraphRAG [local mode]";
python nano_benchmark.py -n 51 -s --mode=local
echo;
echo "Circlemind"
python graph_benchmark.py -n 51 -s

echo "Evaluation of the performance of different RAG methods on the 2wikimultihopqa (101 queries)";
echo;
echo "VectorDB";
python vdb_benchmark.py -n 101 -s
echo;
echo "LightRAG [local mode]";
python lightrag_benchmark.py -n 101 -s --mode=local
echo;
echo "GraphRAG [local mode]";
python nano_benchmark.py -n 101 -s --mode=local
echo;
echo "Circlemind";
python graph_benchmark.py -n 101 -s
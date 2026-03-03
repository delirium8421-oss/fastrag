@echo off
echo Evaluation of the performance of different RAG methods on 2wikimultihopqa (51 queries)
echo.
echo VectorDB
python vdb_benchmark.py -n 51 -s
echo.
echo LightRAG
python lightrag_benchmark.py -n 51 -s --mode=local
echo.
echo GraphRAG
python nano_benchmark.py -n 51 -s --mode=local
echo.
echo Circlemind
python graph_benchmark.py -n 51 -s

echo.
echo.
echo Evaluation of the performance of different RAG methods on 2wikimultihopqa (101 queries)
echo.
echo VectorDB
python vdb_benchmark.py -n 101 -s
echo.
echo LightRAG
python lightrag_benchmark.py -n 101 -s --mode=local
echo.
echo GraphRAG
python nano_benchmark.py -n 101 -s --mode=local
echo.
echo Circlemind
python graph_benchmark.py -n 101 -s

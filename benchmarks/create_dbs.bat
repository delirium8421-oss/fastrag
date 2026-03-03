:: 2wikimultihopqa benchmark
:: Creating databases
python vdb_benchmark.py -n 51 -c
python vdb_benchmark.py -n 101 -c
python lightrag_benchmark.py -n 51 -c
python lightrag_benchmark.py -n 101 -c
python nano_benchmark.py -n 51 -c
python nano_benchmark.py -n 101 -c
python graph_benchmark.py -n 51 -c
python graph_benchmark.py -n 101 -c

:: Evaluation (create reports)
python vdb_benchmark.py -n 51 -b
python vdb_benchmark.py -n 101 -b
python lightrag_benchmark.py -n 51 -b --mode=local
python lightrag_benchmark.py -n 101 -b --mode=local
python nano_benchmark.py -n 51 -b --mode=local
python nano_benchmark.py -n 101 -b --mode=local
python graph_benchmark.py -n 51 -b
python graph_benchmark.py -n 101 -b
# Knowledge Editing for Multi-Hop Question Answering Using Semantic Analysis

This is the repository for [Knowledge Editing for Multi-Hop Question Answering Using Semantic Analysis](), which will be published at the 2025 International Joint Conference on Artificial Intelligence (IJCAI 2025). 

A pre-print has been submitted to arXiv, but is not yet available. Until then the pre-print or conference proceedings are available, the manuscript PDF is provided in this repository.

## CHECK

The MQuAKE datasets are provided in the ```/datasets/``` folder. All CHECK prompts are provided in the ```/prompts/``` folder. Embedding functions used by CHECK are in ```contriever_function.py```.  The relationship typing used by CHECK for the MQuAKE dataset is in ```mquake_relationships.txt```. Other typing files are need to be created for other datasets. The CHECK Knowledge Editing framework is provided in ```CHECK.py```. A working example of CHECK is given in ```CHECK_example.py```.

The following command will run the example script:
``` commandline 
python3 CHECK_example.py --model_name gpt-j --ds_name mquake-cf-3k --ds_start 0 --ds_end 10 --num_new_tokens 50 --similarity cos --sim_thresh 0.8 --use_gpu --gpu_num 0 --verbose
```
Explanations of all command line arguments are provided in example script. The values for ```num_new_tokens```, ```similarity```, and ```sim_thresh``` are the recommended parameters for experimental evaluation.

## Bugs and Questions
If you come across any bugs or issues with the code, or would just like to ask questions about CHECK or generally discuss Knowledge Editing, please feel free to email Dominic Simon at **dominic.simon@ufl.edu** or open an issue on this repository. 

<!--## Citation
If you use our code in your research, please cite our work:
```bibtex
@article{simon2025check,
  title={Knowledge Editing for Multi-Hop Question Answering Using Semantic Analysis},
  author={Simon, Dominic and Ewetz, Rickard},
  journal={International Joint Conference on Artificial Intelligence},
  year={2025}
}
```
-->

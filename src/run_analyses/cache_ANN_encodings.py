"""
Script for taking an input file and caching the activations using a pretrained ANN model.
Used for the search approach.
"""

import argparse
import pandas as pd

from BrainClasses import ANNEncoder #*

def main(raw_args=None):
    parser = argparse.ArgumentParser(description='caching activations')
    # ANN (source) specific
    parser.add_argument('--input_file', type=str, help='Path to input file')
    parser.add_argument('--source_model', default='gpt2-xl', type=str, help='Pretrained model name')
    parser.add_argument('--sent_embed', default='last-tok', type=str, help='How to obtain sentence embeddings')
    parser.add_argument('--actv_cache_setting', default='auto', type=str, help='Which cache setting to use')
    parser.add_argument('--actv_cache_path', 
                        default='/nese/mit/group/evlab/u/asathe/beta-control-neural', 
                        type=str, 
                        help='Where should we cache stuffs?'
                       )

    args = parser.parse_args(raw_args)


    df = pd.read_csv(args.input_file, index_col=[0])

    ####### ANN ENCODER ########
    ann = ANNEncoder(source_model=args.source_model,
                     sent_embed=args.sent_embed,
                     actv_cache_setting=args.actv_cache_setting,
                     actv_cache_path=args.actv_cache_path)

    ann.encode(stimset=df,                                                  
               cache_new_actv=True,                                              
               case=None,                                                        
               **{'stimsetid_suffix':f''},                                       
               include_special_tokens=True,                                      
               verbose=False)  
    
if __name__ == '__main__':
    main()

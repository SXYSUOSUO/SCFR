# Copyright (c)
import pdb
import random

class MultiDataset():
    
    
    def __init__(self, datasets):
        super(MultiDataset, self).__init__()
        # sort indices for reproducible results
        #pdb.set_trace()
        self.datasets = datasets

    def __getitem__(self, idx):
        results=()
        seed=random.randrange(500)
    
        for dataset in self.datasets:
            #pdb.set_trace()
            random.seed(seed)
            #print (self.get_img_info(idx),dataset[idx][1])
            results = results + dataset[idx]
        
        return results


    def get_img_info(self, index):
        #[img_data]
        #results=[]
        #for dataset in self.datasets:
        #    results.append(dataset.get_img_info)
        #return results
        return self.datasets[0].get_img_info(index)
      
    def __len__(self):
        return len(self.datasets[0])

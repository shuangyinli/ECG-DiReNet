from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
   
def visualization(ori_data, generated_data, analysis, name,title):
    """Using PCA or tSNE for generated and original data visualization.
    
    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: 'tsne' or 'pca'
      - name: name identifier for saving plots
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min([10000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
      
    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  
    
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
    
    no, seq_len,feautre = ori_data.shape  
    print(f"ori_data shape: {ori_data.shape}, generated_data shape: {generated_data.shape}")
    
    prep_data = []
    prep_data_hat = []
    
    for i in tqdm(range(anal_sample_no), desc="Processing samples"):
        mean_ori = np.mean(ori_data[i, :, :], axis=1).reshape(1, seq_len)
        mean_gen = np.mean(generated_data[i, :, :], axis=1).reshape(1, seq_len)
        prep_data.append(mean_ori)
        prep_data_hat.append(mean_gen)
    
    prep_data = np.vstack(prep_data)
    prep_data_hat = np.vstack(prep_data_hat)
    
    # Visualization parameter        
    colors = ["red"] * anal_sample_no + ["blue"] * anal_sample_no    
    
    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)
        
        # Plotting
        plt.figure(figsize=(10, 6))    
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], 
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")
      
        plt.legend()  
        plt.title('PCA Plot')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.savefig(f'../score/Generator/Tsne/{name}_pca.png')   
        plt.close() 
    
    elif analysis == 'tsne':
        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
        
        # TSNE analysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
        tsne_results = tsne.fit_transform(prep_data_final)
          
        # Plotting
        plt.figure(figsize=(12, 8))
        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1], 
                    c=colors[:anal_sample_no], alpha=0.2, label="Real")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1], 
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")
      
        plt.legend(fontsize=20) 
          
        plt.title(f'{name} t-SNE',fontsize=30)
        plt.xlabel('')
        plt.ylabel('')
        plt.tight_layout()


        plt.savefig(f'../score/Generator/Tsne/{title}_tsne.pdf')  
        plt.close()   


  
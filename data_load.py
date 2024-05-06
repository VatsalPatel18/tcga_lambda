import os
import pandas as pd

class BasicPreProcess:
    def __init__(self, cancer_type,out_loc=None):
        self.cancer_type = cancer_type
        self.out_dir = f'{out_loc}/{self.cancer_type}'
        self.preprocess_gene_expression()
        self.preprocess_normal_gene_expression()
        self.preprocess_cna_data()
        self.preprocess_methylation_data()
        self.preprocess_mutation_data()

    def preprocess_gene_expression(self):
        path = f'TCGA/Extract/{self.cancer_type}_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt'
        ge_df = (pd.read_csv(path, delimiter='\t', low_memory=False)
                 .dropna()
                 .assign(Hugo_Symbol=lambda df: df['Hugo_Symbol'].str.upper())
                 .rename(columns={'Hugo_Symbol': 'GENES'})
                 .drop_duplicates()
                 .set_index(['GENES', 'Entrez_Gene_Id'])
                 )

        # out_dir = f'{sefl.out_loc}/{self.cancer_type}'
        os.makedirs(self.out_dir, exist_ok=True)
        ge_df.to_csv(f'{out_dir}/ge_df.csv')

    def preprocess_normal_gene_expression(self):
        base_path = '/home/gp7/adv_ml/TCGA/Extract'
        path = f'{base_path}/{self.cancer_type}_tcga_pan_can_atlas_2018/normals/data_mrna_seq_v2_rsem_normal_samples.txt'
        normal_ge_df = (
            pd.read_csv(path, delimiter='\t', low_memory=False)
            .dropna()
            .assign(Hugo_Symbol=lambda df: df['Hugo_Symbol'].str.upper())
            .rename(columns={'Hugo_Symbol': 'GENES'})
            .drop_duplicates()
            .set_index(['GENES', 'Entrez_Gene_Id'])
        )

        os.makedirs(self.out_dir, exist_ok=True)
        normal_ge_df.to_csv(f'{out_dir}/normal_ge_df.csv')

    def preprocess_cna_data(self):
        path = f'TCGA/Extract/{self.cancer_type}_tcga_pan_can_atlas_2018/data_cna.txt'
        cna_df = (
            pd.read_csv(path, delimiter='\t', low_memory=False)
            .dropna()
            .assign(
                Entrez_Gene_Id=lambda df: df['Entrez_Gene_Id'].astype(int),
                Hugo_Symbol=lambda df: df['Hugo_Symbol'].str.upper()
            )
            .rename(columns={'Hugo_Symbol': 'GENES'})
            .drop_duplicates()
        )

        os.makedirs(self.out_dir, exist_ok=True)
        cna_df.to_csv(f'{self.out_dir}/cna_df.csv', index=False)

    def preprocess_methylation_data(self):
        path = f'TCGA/Extract/{self.cancer_type}_tcga_pan_can_atlas_2018/data_methylation_hm27_hm450_merged.txt'
        df = pd.read_csv(path, delimiter='\t', low_memory=False)
        del df['TRANSCRIPT_ID']
        df = df.dropna()
        df['NAME'] = df['NAME'].str.upper()
        df_melted = df.melt(id_vars=['ENTITY_STABLE_ID', 'NAME', 'DESCRIPTION'],
                             var_name='PatientID', value_name='Value')
        df_pivoted = df_melted.pivot_table(index=['PatientID', 'ENTITY_STABLE_ID', 'NAME'],
                                           columns='DESCRIPTION', values='Value', fill_value=0).reset_index()
        df_pivoted.rename(columns={'ENTITY_STABLE_ID': 'Entrez_Gene_Id', 'GENES': 'Entrez_Gene_Id'}, inplace=True)

        os.makedirs(self.out_dir, exist_ok=True)
        df_pivoted.to_csv(f'{self.out_dir}/meth_pivot.csv', index=False)

    def preprocess_mutation_data(self):
        path = f'TCGA/Extract/{self.cancer_type}_tcga_pan_can_atlas_2018/data_mutations.txt'
        mut_df = pd.read_csv(path, delimiter='\t', low_memory=False)
        mut_df = mut_df.dropna()
        mut_df['Entrez_Gene_Id'] = mut_df['Entrez_Gene_Id'].astype(int)
        mut_df = mut_df[['Tumor_Sample_Barcode', 'Hugo_Symbol', 'Entrez_Gene_Id', 'Variant_Classification']]
        mut_df['Hugo_Symbol'] = mut_df['Hugo_Symbol'].str.upper()
        mut_df = mut_df.rename(columns={'Hugo_Symbol': 'GENES', 'Tumor_Sample_Barcode': 'PatientID'})
        encoded_df = pd.get_dummies(mut_df, columns=['Variant_Classification'])
        encoded_df = encoded_df.rename(columns={col: col[23:] for col in encoded_df.columns[3:]})
        encoded_df.columns = [col.split('_')[-1] if 'Variant_Classification' in col else col for col in encoded_df.columns]
        encoded_df2 = encoded_df.groupby(['PatientID', 'GENES', 'Entrez_Gene_Id']).sum().reset_index()

        os.makedirs(self.out_dir, exist_ok=True)
        encoded_df2.to_csv(f'{self.out_dir}/mut_encoded_df.csv', index=False)

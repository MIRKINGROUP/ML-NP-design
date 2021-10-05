import numpy as np
import GPy
import pandas as pd
import itertools
import io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from pymatgen.core.composition import Composition

from camd.agent.base import HypothesisAgent

class EmbedCompGPUCB(HypothesisAgent):
    """
    Acquisition agent for GP-UCB methodology developed by Wahl et al. for discovery 
    of nanoparticles with targeted interface counts that transforms
    the composition space to an embedding space for more effective optimization.
    """

    def __init__(
        self,
        candidate_data=None,
        seed_data=None,
        n_query=None,
        beta='auto',
        kernel=None,
        input_dim=20,
        **kwargs
    ):
        """
        Args:
            candidate_data (pandas.DataFrame): data about the candidates to search over. Must have a "target" column,
                    and at least one additional column that can be used as descriptors.
            seed_data (pandas.DataFrame):  data which to fit the Agent to.
            n_query (int): number of queries in allowed. Defaults to 1.
            beta (float or str): mixing parameter (beta**0.5) for uncertainties in GP-UCB. If a float is given, agent will
                use the same constant beta throughout the campaign. Defaults to 1.0. Setting this as 'auto' will
                use the Theorem 1 from Srivanasan et al. to determine the beta during batch design.
                'auto' has two parameters, 'delta' and 'premultip', which default to 0.1 and 0.05 respectively,
                but can be modified by passing as kwargs to the agent. if mode is "naive" this can only be a flaot.
            kernel (GPy kernel): Kernel object for the GP. Defaults to RBF.
            input_dim (int): dimensionality of the embedding space generated with PCA. Should be smaller than the 
                dimensions of a featurized composition vector.
        """
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query if n_query else 1
        self.beta = beta
        self.kernel = kernel
        self.input_dim = input_dim
        self.kwargs = kwargs
        super(EmbedCompGPUCB).__init__()
        

    def get_hypotheses(self, candidate_data, seed_data=None):
        """
        Methods for getting hypotheses according to the EmbedCompGPUCB algorithm

        Args:
            candidate_data (pandas.DataFrame): candidate data
            seed_data (pandas.DataFrame): seed data

        Returns:
            (pandas.DataFrame): selected hypotheses

        """
        self.candidate_data = candidate_data.drop(
            columns=["target"], axis=1, errors="ignore"
        )

        if seed_data is not None:
            self.seed_data = seed_data
        else:
            raise ValueError("EmbedCompGPUCB requires a finite seed as input. ")

        fb_start = max(len(self.seed_data), 1)

        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=self.input_dim)

        X_seed = self.seed_data.drop(columns=["target"], axis=1)
        y_seed = self.seed_data["target"].to_numpy().reshape(-1, 1)

        y_m, y_std = np.mean(y_seed), np.std(y_seed)
        y_seed = (y_seed - y_m) / y_std

        scaler = StandardScaler()
        scaler.fit(X_seed)
        pca = PCA(self.input_dim)
        pca.fit(scaler.transform(X_seed))
        self.scaler = scaler
        self.pca = pca

        r_seed, r_y, r_candidates = X_seed, y_seed, self.candidate_data

        batch = []
        for i in range(min(self.n_query, len(self.candidate_data))):
            x = pca.transform(scaler.transform(r_seed).astype(np.float64))
            y = r_y.astype(np.float64).reshape(-1, 1)

            m = GPy.models.GPRegression(
                x,
                y,
                kernel=self.kernel,
                noise_var=self.kwargs.get("noise_var", 1.0),
            )
            m.optimize(
                optimizer=self.kwargs.get("optimizer", "bfgs"),
                max_iters=self.kwargs.get("max_iters", 1000),
            )
            self.kernel = m.kern
            self.model = m
            y_pred, var = m.predict(
                pca.transform(scaler.transform(r_candidates.to_numpy().astype(np.float64)))
            )
            t_pred = y_pred * y_std + y_m
            unc = np.sqrt(var) * y_std
            self.t_pred = t_pred
            self.unc = unc

            if self.beta == "auto":
                _t = i + fb_start
                beta = self.kwargs.get("premultip", 0.05) * np.sqrt(
                    2
                    * np.log(
                        len(self.candidate_data)
                        * _t ** 2
                        * np.pi ** 2
                        / 6
                        / self.kwargs.get("delta", 0.1)
                    )
                )
                print("- beta**0.5:{}: ".format(i), beta)
            else:
                beta = self.beta

            t_pred += unc * beta
            s = np.argmax(t_pred)

            name = r_candidates.index.tolist()[s]
            batch.append(name)
            r_seed = r_seed.append(r_candidates.loc[name])
            r_y = np.append(r_y, np.array([y_pred[s]]).reshape(1, 1), axis=0)
            r_candidates = r_candidates.drop(name)

        return self.candidate_data.loc[batch]


# Helper methods used in NP campaigns.
def get_comps(row):
    return Composition(''.join([i[0].replace('%','')+
                                str(i[1]) for i in 
                                list(row.drop(['target','Phases','Interfaces'], errors='ignore').to_dict().items())]))


def get_stoichiometric_formulas(n_components, npoints=6):
    """
    Generates anonymous stoichiometric formulas for a set
    of n_components with specified coefficients

    Args:
        n_components (int): number of components (dimensions)
        grid (list): a range of integers

    Returns:
        (list): unique stoichiometric formula from an
            allowed grid of integers.
    """
    grid = np.linspace(0,1,npoints)
    args = [grid for _ in range(n_components-1)]
    stoics = np.array(list(itertools.product(*args)))
    stoics = stoics[ stoics.sum(axis=1) <= 1.0 ]
    stoics = np.hstack((stoics,1-stoics.sum(axis=1).reshape(-1,1)))
    return stoics

def compare_to_seed(suggestions, df):
    for j, suggestion in suggestions.iterrows():
        i=np.argmin(np.abs(suggestion-df[suggestion.index]).sum(axis=1))
        print(pd.DataFrame.from_records([suggestion.to_dict(),df[list(suggestion.index)+['target']].iloc[i].to_dict()],
                                        index=['suggested','inseed']))
        
def load_np_data(new_raw_data, round_number, elements):
    f = io.StringIO(new_raw_data)
    new_data_df = pd.read_csv(f, sep='\t')/100.0
    for el in elements+['Pt%']:
        if el not in new_data_df.columns:
            new_data_df[el] = 0.0
    new_data_df.index = ['mirkin_r{}_'.format(round_number)+str(i) for i in range(len(new_data_df))]
    new_data_df = new_data_df.round(2)
    new_data_df = new_data_df[~new_data_df.duplicated()]
    return new_data_df


def update_with_new_data(suggestions, new_raw_data, seed_df, seed_data, candidates, 
                         candidate_feats, round_number, elements, measured=0):
    from matminer.featurizers.composition import ElementProperty

    new_data_df = load_np_data(new_raw_data,round_number,elements)
    _elts = ['Au%', 'Ag%', 'Cu%', 'Co%', 'Ni%', 'Pt%', 'Pd%', 'Sn%']
    if measured==0: # all in agreement with SINP target!
        new_data_df['target'] = 0.0
    else:
        new_data_df['target'] = measured
       
    new_data_df['Composition'] = new_data_df.apply(get_comps,axis=1)
    ep = ElementProperty.from_preset(preset_name='magpie')
    new_data_feats = ep.featurize_dataframe(new_data_df, 'Composition').drop(elements+['Pt%','Composition'],axis=1)
    
    seed_df = seed_df[ new_data_df.columns].append(new_data_df)
    seed_data=seed_data.append(new_data_feats) 

    candidates = candidates.drop(suggestions.index)
    candidate_feats = candidate_feats.drop(suggestions.index)
    
    for ind,row in new_data_df[_elts].iterrows():
        candidates = candidates[_elts][ np.any(np.abs(row-candidates)>=0.05,axis=1) ]
    candidate_feats = candidate_feats.loc[candidates.index]
    return seed_df, seed_data, candidates, candidate_feats
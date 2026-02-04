#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:50:29 2025

@author: mac
"""

def j(str_b_i, str_c_i):
    str_b_l = str_b_i.split()
    str_b_l_set = set(str_b_l)
    str_c_l = str_c_i.split()
    str_c_l_set = set(str_c_l)
    the_i = str_b_l_set.intersection(str_c_l_set)
    the_u = str_b_l_set.union(str_c_l_set)
    j_i = len(the_i) / len(the_u)
    return j_i

def word_fun(str_in):
    str_in = clean_txt(str_in)
    tmp = str_in.split()
    the_dictionary = dict()
    for word in set(tmp):
        the_dictionary[word] = tmp.count(word)
    return the_dictionary

def clean_txt(s_in):
    import re
    cln_text = re.sub("[^A-Za-z']+", " ", s_in).strip().lower()
    #cln_text = cln_text.strip()
    #cln_text = cln_text.lower()
    return cln_text

def file_reader(p_in):
    f = open(p_in, "r")
    text = f.read()
    text = clean_txt(text)
    #print (text)
    f.close()
    return text

def file_walker(p_in):
    import os
    import pandas as pd
    main_pd = pd.DataFrame()
    for root, dirs, files in os.walk(p_in):
        for file in files:
            try:
                tmp = file_reader(root + "/" + file)
                if len(tmp) != 0:
                    tmp_pd = pd.DataFrame(
                        {"body": tmp, "label": root.split("/")[-1:][0]},
                        index=[0])
                    main_pd = pd.concat([main_pd, tmp_pd], ignore_index=True)
            except:
                print ("Can't open", root + "/" + file)
                pass
    return main_pd

def rem_sw(c_in):
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    #test_c = "you aren't going to skip class tonight"
    n_str = list() #[]
    for word in c_in.split():
        if word not in sw:
            n_str.append(word)
    o_str = ' '.join(n_str)
    # #in-line looping statement
    # n_str = [word for word in test_c.split() if word not in sw]
    # o_str = ' '.join(n_str)
    return o_str

def word_all_fun(df_in, col_n):
    #https://docs.python.org/3/library/collections.html
    import collections
    the_dictionary = dict()
    all_text = df_in[col_n].str.cat(sep=" ")
    the_dictionary["all"] = collections.Counter(all_text.split())
    
    """
    Expand the above dictionary and create keys for 
    fishing, hiking, machinelearning, mathematics
    """
    for topic in df_in["label"].unique():
        tmp_txt = df_in[df_in["label"] == topic]
        tmp_txt = tmp_txt[col_n].str.cat(sep=" ")
        the_dictionary[topic] = collections.Counter(tmp_txt.split())
    return the_dictionary

def cnt_tok(str_in, sw_in):
    if sw_in == "list":
        t = len(str_in.split())
    else:
        t = len(set(str_in.split()))
    return t

def stem_fun(str_i, sw_i):
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    if sw_i == "porter":
        ps = PorterStemmer()
    else:
        ps = WordNetLemmatizer()
    t_x = list()
    for word in str_i.split():
        if sw_i == "porter":
            t_x.append(ps.stem(word))
        else:
            t_x.append(ps.lemmatize(word))
    text_x = ' '.join(t_x)
    return text_x

def read_pickle(path_in, name_in):
    import pickle
    the_data_t = pickle.load(
        open(path_in + name_in + ".pk", "rb"))
    return the_data_t

def write_pickle(obj_in, path_in, name_in):
    import pickle
    pickle.dump(obj_in, open(
        path_in + name_in + ".pk", "wb"))
    
def xform_fun(df_in, col_n, m_in, n_in, o_path, name_i):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    import pandas as pd
    if name_i == "tf":
        cv = CountVectorizer(ngram_range=(m_in, n_in))
    else:
        cv = TfidfVectorizer(ngram_range=(m_in, n_in))
    x_data_i = pd.DataFrame(
        cv.fit_transform(df_in[col_n]).toarray()) #memory be careful
    x_data_i.columns = cv.get_feature_names_out()
    try:
        x_data_i.index = df_in["label"]
    except:
        pass
    write_pickle(cv, o_path, name_i)
    return x_data_i

def cos_fun(d_a, d_b):
    #turn into a function cos_fun
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    cos_sim = pd.DataFrame(cosine_similarity(d_a, d_b))
    cos_sim.index = d_a.index
    cos_sim.columns = d_a.index
    return cos_sim

def clust_fun(df_in, n_c, o_p):
    #num_clusters = n_c
    from sklearn.cluster import KMeans
    import pandas as pd
    kmeans_model = KMeans(
        n_clusters=n_c, random_state=42, n_init='auto',
        algorithm="elkan")
    kmeans_model.fit(df_in)
    cluster_assignments = pd.DataFrame(kmeans_model.labels_)
    cluster_assignments.columns = ["cluster"]
    write_pickle(kmeans_model, o_p, "cluster")
    return cluster_assignments

def cluster_stats(c_in):
    c_dict = dict()
    for c in set(c_in.index):
        tmp = c_in[c_in.index == c]
        tmp = tmp.groupby("cluster").agg(total=("cluster", "count"))
        c_dict[c] = tmp
    
    clust_dict = dict()
    for k in c_dict.keys():
        tmp = c_dict[k].idxmax()
        clust_dict[k] = tmp[0]
    return clust_dict

def extract_embeddings_pre(df_in, out_path_i, name_in):
    #https://code.google.com/archive/p/word2vec/
    #https://pypi.org/project/gensim/
    #pip install gensim
    #name_in = 'models/word2vec_sample/pruned.word2vec.txt'
    import pandas as pd
    from nltk.data import find
    from gensim.models import KeyedVectors
    import pickle
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        for word in var:
            try:
                tmp_arr.append(list(my_model_t.get_vector(word)))
            except:
                pass
        tmp_arr
        return np.mean(np.array(tmp_arr), axis=0)
    word2vec_sample = str(find(name_in))
    my_model_t = KeyedVectors.load_word2vec_format(
        word2vec_sample, binary=False)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model_t, open(out_path_i + "embeddings.pkl", "wb"))
    pickle.dump(tmp_data, open(out_path_i + "embeddings_df.pkl", "wb" ))
    return tmp_data, my_model_t

def domain_train(df_in, path_in, name_in):
    #domain specific
    import pandas as pd
    import gensim
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        for word in var:
            try:
                tmp_arr.append(list(model.wv.get_vector(word)))
            except:
                pass
        tmp_arr
        return np.mean(np.array(tmp_arr), axis=0)
    model = gensim.models.Word2Vec(df_in.str.split())
    model.save(path_in + 'body.embedding')
    #call up the model
    #load_model = gensim.models.Word2Vec.load('body.embedding')
    #model.wv.similarity('fish','river')
    tmp_data = pd.DataFrame(df_in.str.split().apply(get_score))
    tmp_data = tmp_data.apply(lambda x: {
        f"{k}": vv for v in x for k, vv in enumerate(v, 0)
        },result_type="expand", axis=1,
)
    return tmp_data, model

def chi_fun(df_in, lab_in, k_in, p_in, n_in, stat_sig):
    from sklearn.feature_selection import chi2, SelectKBest
    import pandas as pd
    feat_sel = SelectKBest(score_func=chi2, k=k_in)
    dim_data = pd.DataFrame(feat_sel.fit_transform(df_in, lab_in))
    p_val = pd.DataFrame(list(feat_sel.pvalues_))
    p_val.columns = ["pval"]
    feat_index = list(p_val[p_val.pval <= stat_sig].index)
    dim_data = dim_data[feat_index]
    feature_names = df_in.columns[feat_index]
    dim_data.columns = feature_names
    write_pickle(feat_sel, p_in, n_in)
    write_pickle(dim_data, p_in, "chi_data_" + n_in)
    return dim_data, feat_sel

def llm_fun(df_in, p_in, n_in):
    #https://pypi.org/project/sentence-transformers/
    #https://huggingface.co/models?library=sentence-transformers
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    if n_in == "small":
        llm = 'sentence-transformers/all-MiniLM-L6-v2'
    else:
        llm = 'sentence-transformers/all-mpnet-base-v2' #takes time but superb performance
    model = SentenceTransformer(llm)
    write_pickle(model, p_in, n_in)
    vec_t = pd.DataFrame(model.encode(df_in))
    return vec_t

def pca_fun(df_in, e_v, o_p):
    from sklearn.decomposition import PCA
    import pandas as pd
    pca = PCA(n_components=e_v)
    xform_data_t = pd.DataFrame(pca.fit_transform(df_in))
    exp_var = sum(pca.explained_variance_ratio_)
    print ("Explained Variance:", exp_var,
           "Number Components", len(xform_data_t.columns))
    write_pickle(pca, o_p, "pca")
    return xform_data_t

def senti_fun(str_in):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    senti = SentimentIntensityAnalyzer()
    the_senti = senti.polarity_scores(str_in)["compound"]
    return the_senti

def model_fun(df_in, l_in, t_s, sw_in, o_o):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, l_in, test_size=t_s, random_state=42)
    
    if sw_in == "rf":
        mod = RandomForestClassifier(
            max_depth=10, n_estimators=10, random_state=123)
    elif sw_in == "gnb":
        mod = GaussianNB()
    elif sw_in == "gbc":
        mod = GradientBoostingClassifier()
    
    mod.fit(X_train, y_train)
    write_pickle(mod, o_o, sw_in)
    y_pred = mod.predict(X_test)
    y_pred_proba = pd.DataFrame(mod.predict_proba(X_test))
    y_pred_proba.columns = mod.classes_
    m_metrics = pd.DataFrame(precision_recall_fscore_support(
        y_test, y_pred, average='weighted'))
    m_metrics.index = ["precision", "recall", "fscore", None]
    
    f_i = feat_fun(df_in, mod, sw_in, o_o)
    
    # #feature importance
    # try:
    #     feat_imp = pd.DataFrame(mod.feature_importances_)
    #     feat_imp.index = df_in.columns
    #     feat_imp.columns = ["fi"]
    #     non_zero = len(feat_imp[feat_imp["fi"] != 0])
    #     print ("Total % with Propensity", non_zero/len(feat_imp)*100, "%")
    # except:
    #     print (sw_in, "does not support feature importance")
    #     pass
    return mod

def feat_fun(df_in, mod_i, sw_i, o_p):
    import pandas as pd
    try:
        feat_imp = pd.DataFrame(mod_i.feature_importances_)
        feat_imp.index = df_in.columns
        feat_imp.columns = ["fi"]
        non_zero = len(feat_imp[feat_imp["fi"] != 0])
        print ("Total % with Propensity", non_zero/len(feat_imp)*100, "%")
        feat_imp.to_csv(o_p + sw_i + ".csv")
    except:
        print (sw_i, "does not support feature importance")
        pass
    return feat_imp

def grid_fun(df_in, l_in, para_i, t_s, n_f, sw_i, o_p):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, l_in, test_size=t_s, random_state=42)
    
    if sw_i == "rf":
        mod = RandomForestClassifier(random_state=42)
    elif sw_i == "gnb":
        mod = GaussianNB()
    elif sw_i == "gbc":
        mod = GradientBoostingClassifier()
    
    gs = GridSearchCV(estimator=mod, param_grid=para_i, cv=n_f)
    gs.fit(X_train, y_train)
    
    best_perf = gs.best_score_
    params_next = gs.best_params_
    print ("Best score", best_perf, "Params", params_next)
    
    if sw_i == "rf":
        mod = RandomForestClassifier(**gs.best_params_)
    elif sw_i == "gnb":
        mod = GaussianNB(**gs.best_params_)
    elif sw_i == "gbc":
        mod = GradientBoostingClassifier(**gs.best_params_)
        
    mod.fit(df_in, l_in)
    write_pickle(mod, o_p, sw_i)
    
    f_i = feat_fun(df_in, mod, sw_i, o_p)
    f_i.to_csv(o_p + sw_i + "_fi")
    
    return mod
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#计算最大回撤
def cal_maxdrawdown(data):
    if isinstance(data,list):
        data=np.array(data)
    if isinstance(data, pd.Series):
        data=data.values
    
    def get_mdd(values):
        dd=[values[i:].min()/values[i]-1 for i in range(len(values))]
        return abs(min(dd))
    
    if not isinstance(data, pd.DataFrame):
        return get_mdd(data)
    
    else:
        return data.apply(get_mdd)

#计算投资组合指标
def cal_indicators(df_cum_value):
    '''
    功能：给定累计净值，计算各组合的评价指标，包括：年化收益率、年化标准差、夏普值、最大回撤
    输入：
        df_daily_cum_return  pd.DataFrame，index为升序排列的日期，columns为各组合名称，value为daily_cum_return
    '''
    df_daily_return=np.log(df_cum_value/df_cum_value.shift(1)).dropna()
    #df_cum_value = (df_daily_return + 1).cumprod()
    res = pd.DataFrame(index=['年化收益率','年化标准差','夏普值','最大回撤'], columns=df_daily_return.columns, data=0.0)
    res.loc['年化收益率'] = (df_daily_return.mean() * 250).apply(lambda x: '%.2f%%' % (x*100))
    res.loc['年化标准差'] = (df_daily_return.std() * np.sqrt(250)).apply(lambda x: '%.2f%%' % (x*100))
    res.loc['夏普值'] = (df_daily_return.mean() / df_daily_return.std() * np.sqrt(250)).apply(lambda x: np.round(x, 2))
    res.loc['最大回撤'] = cal_maxdrawdown(df_cum_value).apply(lambda x: '%.2f%%' % (x*100))
    return res
    

#获取资产训练集
def get_training_set(start_year,end_year,asset_return_list):
    
    Rtn_list=[]#每个资产的年化收益率
    Vol_list=[]#每个资产的波动率
    for i in range(45):
        Rtn_list.append((asset_return_list[i][str(start_year):str(end_year+1)].mean().values[0])*250)
        Vol_list.append((asset_return_list[i][str(start_year):str(end_year+1)].std().values[0]) * np.sqrt(250))
        
    df_training_set1=pd.DataFrame({'AnnualizedRtn':Rtn_list,'Volatility':Vol_list})
    
    training_set1=[]
    for i in range(45):
        
        training_set1.append(list(df_training_set1.iloc[i])) 
        
    arr_training_set1=np.array(training_set1)
    
    return arr_training_set1
    

#获取无调整K mean聚类结果
def get_static_kmeans_result(N,start_year,end_year,training_set):
    #N为聚类次数
    #start_year和end_year为回测的始末日期
    #training_set为array
    
    model = KMeans(n_clusters=N,init='k-means++').fit(arr_training_set)
    arr_model_centers=model.cluster_centers_
    arr_model_labels=model.labels_
    
    #获取中心资产的index
    index_value=[]
    for i in range(len(arr_model_centers)):
        a=training_set-arr_model_centers[i]
        b= list(np.linalg.norm(a[i]) for i in range(len(a)))
        index_value.append(b.index(min(b)))#返回最小值所在index
    
    #得到各资产收益表data_all
    data_all=pd.DataFrame()#存储聚类下各类资产收益率
    for i in index_value:
        data_all=pd.merge(data_all,data_return[i],how='outer',left_index=True,right_index=True)
    data_all=data_all.fillna(0.00)
    
    
    weights=np.ones((N,1))*1/N
    daily_rtn=data_all.loc[str(start_year):str(end_year+1)]
    daily_rtn.ix[0]=0.0
    
    
    #获取组合累计净值
    df_portfolio_return=((daily_rtn.cumsum()+1).dot(weights))
    df_portfolio_return.columns = ['N='+str(N)]
    
    return df_portfolio_return, index_value
    
    
#每年初动态调整K means结果    
def get_dynamic_kmeans_result(N,start_year,training_set):
    #N为聚类次数
    #start_year和end_year为回测的始末日期
    #training_set为array
    
    model = KMeans(n_clusters=N,init='k-means++').fit(training_set)
    arr_model_centers=model.cluster_centers_
    arr_model_labels=model.labels_
    
    #获取中心资产的index
    index_value=[]
    for i in range(len(arr_model_centers)):
        a=training_set-arr_model_centers[i]
        b= list(np.linalg.norm(a[i]) for i in range(len(a)))
        index_value.append(b.index(min(b)))#返回最小值所在index
    
    #得到各资产收益表data_all
    data_all=pd.DataFrame()#存储聚类下各类资产收益率
    for i in index_value:
        data_all=pd.merge(data_all,data_return[i],how='outer',left_index=True,right_index=True)
    data_all=data_all.fillna(0.00)
    
    
    weights=np.ones((N,1))*1/N
    daily_rtn=data_all.loc[str(start_year)]
    daily_rtn.ix[0]=0.0
    
    
    #获取组合每日收益率
    df_portfolio_return=(daily_rtn.dot(weights))
    df_portfolio_return.columns = ['N='+str(N)]
    
    return df_portfolio_return, index_value
    

data1=pd.read_excel(u'泛欧斯托克600行情统计.xlsx')
data2=pd.read_excel(u'白银行情统计.xlsx')
data3=pd.read_excel(u'黄金行情统计.xlsx')
data4=pd.read_excel(u'布伦特原油行情统计.xlsx')
data5=pd.read_excel(u'澳洲标普200行情统计.xlsx')
data6=pd.read_excel(u'新西兰50行情统计.xlsx')
data7=pd.read_excel(u'墨西哥指数行情统计.xlsx')
data8=pd.read_excel(u'纳斯达克指数行情统计.xlsx')
data9=pd.read_excel(u'标普500行情统计.xlsx')
data10=pd.read_excel(u'道琼斯工业行情统计.xlsx')
data11=pd.read_excel(u'巴西圣保罗指数行情统计.xlsx')
data12=pd.read_excel(u'挪威OSE行情统计.xlsx')
data13=pd.read_excel(u'希腊ASE行情统计.xlsx')
data14=pd.read_excel(u'瑞士SMI行情统计.xlsx')
data15=pd.read_excel(u'布拉格综指行情统计.xlsx')
data16=pd.read_excel(u'丹麦哥本哈根指数行情统计.xlsx')
data17=pd.read_excel(u'比利时BFX行情统计.xlsx')
data18=pd.read_excel(u'奥地利ATX行情统计.xlsx')
data19=pd.read_excel(u'荷兰AEX行情统计.xlsx')
data20=pd.read_excel(u'英国富时100行情统计.xlsx')
data21=pd.read_excel(u'俄罗斯RTS行情统计.xlsx')
data22=pd.read_excel(u'德国DAX行情统计.xlsx')
data23=pd.read_excel(u'法国CAC40行情统计.xlsx')
data24=pd.read_excel(u'芬兰OMX行情统计.xlsx')
data25=pd.read_excel(u'马来西亚吉隆坡指数行情统计.xlsx')
data26=pd.read_excel(u'斯里兰卡科伦坡指数行情统计.xlsx')
data27=pd.read_excel(u'上证综指行情统计.xlsx')
data28=pd.read_excel(u'上证180行情统计.xlsx')
data29=pd.read_excel(u'中证1000行情统计.xlsx')
data30=pd.read_excel(u'中小板指行情统计.xlsx')
data31=pd.read_excel(u'富时新加坡行情统计.xlsx')
data32=pd.read_excel(u'富时巴基斯坦行情统计.xlsx')
data33=pd.read_excel(u'韩国综指行情统计.xlsx')
data34=pd.read_excel(u'日经225行情统计.xlsx')
data35=pd.read_excel(u'印度孟买SENSEX行情统计.xlsx')
data36=pd.read_excel(u'印尼雅加达行情统计.xlsx')
data37=pd.read_excel(u'泰国综指行情统计.xlsx')
data38=pd.read_excel(u'台湾加权行情统计.xlsx')
data39=pd.read_excel(u'越南胡志明行情统计.xlsx')
data40=pd.read_excel(u'深证成指行情统计.xlsx')
data41=pd.read_excel(u'创业板指行情统计.xlsx')
data42=pd.read_excel(u'沪深300行情统计.xlsx')
data43=pd.read_excel(u'上证50行情统计.xlsx')
data44=pd.read_excel(u'菲律宾马尼拉行情统计.xlsx')
data45=pd.read_excel(u'恒生指数行情统计.xlsx')


data_list=[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,
           data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,
           data21,data22,data23,data24,data25,data26,data27,data28,data29,data30,
           data31,data32,data33,data34,data35,data36,data37,data38,data39,data40,
           data41,data42,data43,data44,data45]

for i in range(45):
    data_list[i].set_index(['TradingDay'],inplace=True)
    data_list[i].index=pd.to_datetime(data_list[i].index)

#获取各资产对数收益率
data_return=[]
for i in range(45):
    data_return.append(np.log(data_list[i]/data_list[i].shift(1)).dropna())

#无调整k means结果
arr_training_set=get_training_set(2010,2014,data_return)
#获取静态资产组合净值和index
df_static_kmeans_result=pd.DataFrame()
asset_index=[]
#计算聚类次数从5次到10次的结果
for i in range(6):
    (portfolio_return,index_value)=get_static_kmeans_result(i+5,2015,2018,arr_training_set)
    df_static_kmeans_result=pd.merge(df_static_kmeans_result,portfolio_return,how='outer',left_index=True,right_index=True)
    asset_index.append(index_value)

df_static_kmeans_result=df_static_kmeans_result.fillna(method='pad')
df_static_kmeans_result=df_static_kmeans_result.fillna(method='bfill')

#查看每一种聚类的大类资产种类
asset_index

df_static_kmeans_result.index=pd.to_datetime(df_static_kmeans_result.index)
df_static_result=cal_indicators(df_static_kmeans_result)
df_static_result #获取描述性统计量表格

df_static_kmeans_result.plot(figsize=(15,10)).grid(True)#画图


#动态调整的K means聚类
index_list_all=[]
df_dynamic_kmeans_result=pd.DataFrame()
for i in range(4):
    arr_training_set=get_training_set(2010+i,2014+i,data_return)
    df_static_kmeans_result=pd.DataFrame()
    for d in range(6):
        (portfolio_return,index_value)=get_dynamic_kmeans_result(d+5,2015+i,arr_training_set)
        df_static_kmeans_result=pd.merge(df_static_kmeans_result,portfolio_return,how='outer',left_index=True,right_index=True)
        index_list_all.append(index_value)
    df_static_kmeans_result=df_static_kmeans_result.fillna(method='pad')
    df_static_kmeans_result=df_static_kmeans_result.fillna(method='bfill')
    
    #print df_static_kmeans_result
    df_dynamic_kmeans_result=df_dynamic_kmeans_result.append(df_static_kmeans_result)
    
#获取累计净值数据
df_dynamic_kmeans_result_cum=df_dynamic_kmeans_result.cumsum()+1

#获取聚类的底层资产名称
index_list_n5=[]
index_list_n6=[]
index_list_n7=[]
index_list_n8=[]
index_list_n9=[]
index_list_n10=[]
for i in range(len(index_list_all)):
    if len(index_list_all[i])==5:
        index_list_n5.append(index_list_all[i])
    elif len(index_list_all[i])==6:
        index_list_n6.append(index_list_all[i])
    elif len(index_list_all[i])==7:
        index_list_n7.append(index_list_all[i])
    elif len(index_list_all[i])==8:
        index_list_n8.append(index_list_all[i])
    elif len(index_list_all[i])==9:
        index_list_n9.append(index_list_all[i])
    elif len(index_list_all[i])==10:
        index_list_n10.append(index_list_all[i])

#获取带调整的描述性统计表格
df_dynamic_result=cal_indicators(df_dynamic_kmeans_result_cum)

#画图
df_dynamic_kmeans_result_cum.plot(figsize=(15,10)).grid(True)

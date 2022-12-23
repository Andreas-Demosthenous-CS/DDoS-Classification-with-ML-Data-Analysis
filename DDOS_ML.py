from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from ddos_utils import *


def get_KBest_features(dataset, res):
    print("Dataset shape: "+str(dataset.shape))
    res += str(dataset.shape) + "\n"
    univar_algorithms_names = ['chi2', 'f_classif', 'mutual_info_classif']
    univar_algorithms = [chi2, f_classif, mutual_info_classif]
    features = get_feature_names(dataset)
    K = [5, 10, 15, 20, 25]
    for k in K:
        print("For k = "+str(k))
        res+="k = "+str(k)+"\n"

        for algorithm, algorithmName in zip(univar_algorithms, univar_algorithms_names):
            print("\nRunning algorithm " + str(algorithmName) + ":")
            res+=str(algorithmName)+": "

            results = get_KBestFeatures_Univariate_selection(X, y, algorithm, k)
            k_best = []
            for f in results:
                k_best.append(features[f])
            print(k_best)
            res+=str(k_best)+"\n"

    print("RES: ")
    print(res)

def find_best(accuracy_table):
    best_training_alg = 0
    best_validation_alg = 0
    best_training_accuracy = 0
    best_validation_accuracy = 0

    cnt = 0
    for alg in accuracy_table:
        if(alg[0] > best_training_accuracy):
            best_training_accuracy = alg[0]
            best_training_alg = cnt
        if(alg[1] > best_validation_accuracy):
            best_validation_accuracy = alg[1]
            best_validation_alg = cnt
        cnt+=1
    return best_training_alg, best_training_accuracy, best_validation_alg, best_validation_accuracy

pd.options.mode.use_inf_as_na = True  ## so that inf is also treated as NA value

client = Client(processes=False)

dataset_filenames = ['DrDoS_DNS.csv', 'DrDoS_LDAP.csv', 'DrDoS_MSSQL.csv', 'DrDoS_NetBIOS.csv', 'DrDoS_NTP.csv', 'DrDoS_SNMP.csv', 'DrDoS_SSDP.csv', 'DrDoS_UDP.csv', 'Syn.csv', 'UDPLag.csv']


K_best_5000 = [
    ['FlowDuration', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'BwdIATTotal'],
    ['Protocol', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'Inbound'],
    ['FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean'],

    ['FlowDuration', 'TotalLengthofBwdPackets', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'BwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'Inbound'],
    ['FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FwdHeaderLength', 'Inbound'],

    ['FlowDuration', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FwdPSHFlags', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'Down/UpRatio', 'min_seg_size_forward', 'Inbound'],

    ['FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'FlowDuration', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd','BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean','BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'IdleStd', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'Inbound'],

    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets','FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd','FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength','URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd','Inbound'],
    ['Protocol', 'FlowDuration', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'SYNFlagCount', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'IdleStd', 'Inbound'],

    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalLengthofFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowIATMin', 'FwdIATTotal', 'FwdIATMean', 'FwdIATMax', 'FwdIATMin', 'FwdHeaderLength', 'FwdPackets/s', 'MinPacketLength', 'MaxPacketLength', 'PacketLengthStd', 'ACKFlagCount', 'AveragePacketSize', 'SubflowFwdBytes', 'Init_Win_bytes_forward', 'min_seg_size_forward'],

]

K_best_10000 = [

    ['FlowDuration', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'BwdIATTotal'],
    ['Protocol', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'Inbound'],
    ['FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean'],

    ['FlowDuration', 'TotalLengthofBwdPackets', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'TotalFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'BwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'URGFlagCount', 'Down/UpRatio', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FwdHeaderLength', 'Inbound'],

    ['FlowDuration', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'TotalFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FwdPSHFlags', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'min_seg_size_forward', 'Inbound'],

    ['FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'IdleStd', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'Inbound'],

    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'SYNFlagCount', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'IdleStd', 'Inbound']

    ,['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalLengthofFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowIATMin', 'FwdIATTotal', 'FwdIATMean', 'FwdIATMax', 'FwdIATMin', 'FwdHeaderLength', 'FwdPackets/s', 'MinPacketLength', 'MaxPacketLength', 'PacketLengthStd', 'ACKFlagCount', 'AveragePacketSize', 'SubflowFwdBytes', 'Init_Win_bytes_forward', 'min_seg_size_forward'],

]

K_best_50000 = [

    ['FlowDuration', 'FlowBytes/s', 'FlowIATMean', 'BwdIATTotal', 'IdleStd'],
    ['Protocol', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'Inbound'],
    ['FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean'],

    ['FlowDuration', 'TotalLengthofBwdPackets', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'BwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FwdHeaderLength', 'min_seg_size_forward'],

    ['FlowDuration', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'FlowDuration', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FwdPSHFlags', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'FwdHeaderLength', 'BwdHeaderLength', 'min_seg_size_forward', 'Inbound'],

    ['FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'FlowDuration', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'IdleStd', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'Inbound'],

    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound']

    ,['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalLengthofFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowIATMin', 'FwdIATTotal', 'FwdIATMean', 'FwdIATMax', 'FwdIATMin', 'FwdHeaderLength', 'FwdPackets/s', 'MinPacketLength', 'MaxPacketLength', 'PacketLengthStd', 'ACKFlagCount', 'AveragePacketSize', 'SubflowFwdBytes', 'Init_Win_bytes_forward', 'min_seg_size_forward'],

]

K_best_100000 = [

    ['FlowDuration', 'FlowBytes/s', 'FlowIATMean', 'BwdIATTotal', 'IdleStd'],
    ['Protocol', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'Inbound'],
    ['FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean'],

   ['FlowDuration', 'TotalLengthofBwdPackets', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'ActiveMean', 'ActiveStd', 'IdleStd'],
   ['Protocol', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'BwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'Inbound'],
   ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FwdHeaderLength', 'min_seg_size_forward'],

    ['FlowDuration', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'FlowDuration', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FwdPSHFlags', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'FwdHeaderLength', 'BwdHeaderLength', 'min_seg_size_forward', 'Inbound'],

    ['FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'FlowDuration', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'IdleStd', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'Inbound'],

    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound'],
    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound'],

    ['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalLengthofFwdPackets', 'FwdPacketLengthMax',
     'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowIATMin', 'FwdIATTotal', 'FwdIATMean', 'FwdIATMax', 'FwdIATMin',
     'FwdHeaderLength', 'FwdPackets/s', 'MinPacketLength', 'MaxPacketLength', 'PacketLengthStd', 'ACKFlagCount',
     'AveragePacketSize', 'SubflowFwdBytes', 'Init_Win_bytes_forward', 'min_seg_size_forward']

]

K_best_500000 = [

['FlowDuration', 'FlowBytes/s', 'FlowIATMean', 'BwdIATTotal', 'IdleStd'],
['Protocol', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'Inbound'],
['FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean'],

['FlowDuration', 'TotalLengthofBwdPackets', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'ActiveMean', 'ActiveStd', 'IdleStd'],
['Protocol', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'Inbound'],
['Protocol', 'FlowDuration', 'TotalFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FwdHeaderLength', 'min_seg_size_forward'],

['FlowDuration', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'ActiveMean', 'ActiveStd', 'IdleStd'],
['Protocol', 'FlowDuration', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FwdPSHFlags', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'Inbound'],
['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'FwdHeaderLength', 'BwdHeaderLength', 'min_seg_size_forward', 'Inbound'],

['FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd'],
['Protocol', 'FlowDuration', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'IdleStd', 'Inbound'],
['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound'],

['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd'],
['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound'],
['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound']

,['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalLengthofFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowIATMin', 'FwdIATTotal', 'FwdIATMean', 'FwdIATMax', 'FwdIATMin', 'FwdHeaderLength', 'FwdPackets/s', 'MinPacketLength', 'MaxPacketLength', 'PacketLengthStd', 'ACKFlagCount', 'AveragePacketSize', 'SubflowFwdBytes', 'Init_Win_bytes_forward', 'min_seg_size_forward'],

]

K_best_750000 = [
['FlowDuration', 'FlowBytes/s', 'FlowIATMean', 'BwdIATTotal', 'IdleStd'],
['Protocol', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'Inbound'],
['FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean'],

['FlowDuration', 'TotalLengthofBwdPackets', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'ActiveMean', 'ActiveStd', 'IdleStd'],
['Protocol', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'Inbound'],
['Protocol', 'FlowDuration', 'TotalFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FwdHeaderLength', 'min_seg_size_forward'],

['FlowDuration', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'ActiveMean', 'ActiveStd', 'IdleStd'],
['Protocol', 'FlowDuration', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FwdPSHFlags', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'Inbound'],
['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'FwdHeaderLength', 'BwdHeaderLength', 'min_seg_size_forward', 'Inbound'],

['FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd'],
['Protocol', 'FlowDuration', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'IdleStd', 'Inbound'],
['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound'],

['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd'],
['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'BwdIATTotal', 'FwdPSHFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound'],
['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'BwdPacketLengthMin', 'BwdPacketLengthStd', 'FlowBytes/s', 'FlowPackets/s', 'FlowIATMean', 'FlowIATMin', 'BwdIATTotal', 'FwdHeaderLength', 'BwdHeaderLength', 'URGFlagCount', 'CWEFlagCount', 'Down/UpRatio', 'min_seg_size_forward', 'ActiveMean', 'ActiveStd', 'IdleStd', 'Inbound']

,['Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalLengthofFwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthStd', 'FlowIATMin', 'FwdIATTotal', 'FwdIATMean', 'FwdIATMax', 'FwdIATMin', 'FwdHeaderLength', 'FwdPackets/s', 'MinPacketLength', 'MaxPacketLength', 'PacketLengthStd', 'ACKFlagCount', 'AveragePacketSize', 'SubflowFwdBytes', 'Init_Win_bytes_forward', 'min_seg_size_forward'],

]

K_best_custom = [
    ['Protocol','FlowDuration','TotalFwdPackets','TotalLengthofFwdPackets','FwdPacketLengthMax','FwdPacketLengthMin','FwdPacketLengthStd', 'FlowIATMin', 'FwdIATTotal','FwdIATMean','FwdIATMax','FwdIATMin','FwdHeaderLength','FwdPackets/s','MinPacketLength','MaxPacketLength','PacketLengthStd','ACKFlagCount','AveragePacketSize','SubflowFwdBytes','Init_Win_bytes_forward','min_seg_size_forward'],
]

res = ""

F = ['Unnamed:0', 'FlowID', 'SourceIP', 'DestinationIP', 'Timestamp', 'SimillarHTTP', 'SourcePort', 'DestinationPort']
G = ['BwdPSHFlags', 'FwdURGFlags', 'BwdURGFlags', 'FINFlagCount', 'PSHFlagCount', 'ECEFlagCount', 'FwdAvgBytes/Bulk', 'FwdAvgPackets/Bulk', 'FwdAvgBulkRate', 'BwdAvgBytes/Bulk', 'BwdAvgPackets/Bulk', 'BwdAvgBulkRate']
H = ['ActiveMin', 'AveragePacketSize', 'FwdHeaderLength.1', 'MinPacketLength', 'SubflowBwdBytes', 'BwdIATMax', 'ACKFlagCount', 'ActiveMax', 'BwdPacketLengthMean', 'BwdIATMean', 'RSTFlagCount', 'SubflowBwdPackets', 'AvgFwdSegmentSize', 'FwdPacketLengthMean', 'MaxPacketLength', 'PacketLengthVariance', 'FwdIATStd', 'BwdPacketLengthMax', 'Init_Win_bytes_forward', 'PacketLengthMean', 'BwdIATMin', 'FwdPackets/s', 'FwdIATMin', 'IdleMin', 'BwdPackets/s', 'BwdIATStd', 'IdleMean', 'FwdIATMax', 'FlowIATMax', 'AvgBwdSegmentSize', 'IdleMax', 'SubflowFwdPackets', 'Init_Win_bytes_backward', 'FlowIATStd', 'FwdIATTotal', 'TotalLengthofFwdPackets', 'FwdIATMean', 'SubflowFwdBytes', 'PacketLengthStd', 'act_data_pkt_fwd']

i = 0
models = []
models.append(('Decision Tree - DT',  DecisionTreeClassifier()))
models.append(('Random Forest - RF',make_pipeline(StandardScaler(),  RandomForestClassifier())))
models.append(('K-Nearest Neigbors - KNN',make_pipeline(StandardScaler(), KNeighborsClassifier(leaf_size=11))))
models.append(('Extreme Gradient Boosting Classifier - XGBC',XGBClassifier()))
models.append(('Naive Bayes - NB',make_pipeline(StandardScaler(),  GaussianNB())))
models.append(
    ('Neural Network -- Multi-layer Perceptron classifier - MLP', make_pipeline(StandardScaler(),MLPClassifier(random_state=1, max_iter=10000))))
models.append(('Logistic Regression - LR', make_pipeline(StandardScaler(), LogisticRegression(solver='saga', max_iter=700))))
models.append(('Linear Discriminant Analysis - LDA', make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())))

univar_algorithms_names = ['chi2', 'f_classif', 'mutual_info_classif']

sample_amounts = [5000, 10000, 50000]
K_Best_All = [K_best_5000, K_best_10000, K_best_50000, K_best_100000]

for sample_amount in sample_amounts:
    print("Sample amount = " + str(sample_amount))
    res = res + str(sample_amount)+"\n"
    dir_to_make = "sample_" + str(sample_amount)
    outputfile = dir_to_make +"\\results_sample_" + str(sample_amount) + ".txt"

    path = "C:\\Users\\andre\\PycharmProjects\\ML-test\\"
    path = os.path.join(path, dir_to_make)
    if (not os.path.isdir(path)):
        os.mkdir(path)
        sys.stdout = Logger(outputfile)
        print("\nReducing initial dataset: ")
        reducing_initial_dataset(dir_to_make, sample_amount, dataset_filenames, 0)

        print("\nMerging partial datasets: ")
        merging_partial_datasets(dir_to_make, dataset_filenames)
    else:
        sys.stdout = Logger(outputfile)

    print("\nLoading reduced dataset: ")

    dataset_file = dir_to_make+"\\reduced_dataset.csv"
    dataset = readfile(dataset_file)

    dataset.dropna(inplace=True)
    print("Removing irrelevant features: ")
    print(F)
    dataset.drop(columns=F, axis=1, inplace =True)

    #print("Removing constant features: ")
    #print(G)
    #dataset.drop(columns=G, axis=1, inplace =True)

    #print("Removing highly correlated features: ")
    #print(H)
    #dataset.drop(columns=H, axis=1, inplace =True)
    # dataset.drop(columns=H, axis=1, inplace=True)


    #replacing negative values with 0
    dataset[dataset<0] = 0

    dataset = dataset.sample(frac=1)
    dataset = dataset[dataset.Label <= 10]
    #dataset[dataset.Label > 0] = 1

    # X -> all columns except the target column(Label)
    X = dataset.drop(columns=["Label"], axis=1)
    # y -> LabelEncoded column: target column
    y = dataset[["Label"]]


    print(" DDoS class distribution:")
    print(dataset.groupby("Label").size())


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, stratify=y, random_state=45)

    cnt = 0
    accuracy_times_sum = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    #for k_best_subset in K_Best_All[i]:
    for k_best_subset in K_best_custom:

        X_train_k_best = X_train[k_best_subset]
        X_val_k_best = X_val[k_best_subset]
        print("\n\nX = "+str(sample_amounts[i])+", K-BEST SUBSET: "+str(cnt+1)+" Algorithm: "+univar_algorithms_names[cnt%3])
        print("----------------------------------------------------------------------")
        print("K-Best subset: ")
        print(k_best_subset)
        print("----------------------------------------------------------------------")
        accuracy_times_sum = run_ML_algorithms(X_train_k_best, X_val_k_best, y_train, y_val, models, accuracy_times_sum, cnt+1, len(k_best_subset))

        best_training_alg, best_training_accuracy , best_validation_alg , best_validation_accuracy  = find_best(accuracy_times_sum)
        print("----------------------------------------------------------------------")
        print("Most Accurate training algorithm: "+models[best_training_alg][0]+",Mean Accuracy: %0.3f" % (best_training_accuracy/(cnt+1))+", Subset num = "+
                ", Most Accurate validation algorithm: "+models[best_validation_alg][0]+", Mean Accuracy: %0.3f" %(best_validation_accuracy/(cnt+1)))
        print("----------------------------------------------------------------------")
        print("Algorithms stats: ")

        alg_index = 0
        for alg in accuracy_times_sum:
            """
             print("Algorithm: "+models[alg_index][0]+", Mean training duration: %0.3f" %(alg[2]/(cnt+1))+", Mean Training accuracy: %0.3f" %(alg[0]/(cnt+1))+", Best Training Accuracy: %0.3f" %(alg[3])+", Subset = "+str(alg[4])+", k = "+str(alg[5])+
                    ", Mean Validation Accuracy: %0.3f" %(alg[1]/(cnt+1))+", Best Validation Accuracy: %0.3f" %(alg[6])+", Subset = "+str(alg[7])+", k = "+str(alg[8]))
             """
            print("Algorithm: " + models[alg_index][0] + "\n%0.3f" % (
                        alg[2] / (cnt + 1)) + "\n%0.3f" % (
                              alg[0] / (cnt + 1)) + "\n%0.3f" % (alg[3]) + "\n( S = " + str(
                alg[4]) + ", k = " + str(alg[5]) +
                  " )\n%0.3f" % (alg[1] / (cnt + 1)) + "\n%0.3f" % (
                  alg[6]) + "\n( S = " + str(alg[7]) + ", k = " + str(alg[8])+" )\n")
            alg_index+=1
        cnt+=1
    i+=1
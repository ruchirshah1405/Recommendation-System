from pyspark.mllib.recommendation import ALS, Rating
from pyspark import SparkContext
from pyspark.mllib.evaluation import RegressionMetrics
import time
import sys
import math

st = time.time()
sc = SparkContext('local[*]','NAME')

def getprediction(data):
    sum1 = 0
    if data[0] not in trainUserRDD.keys():
        return ((data[0],data[1]),3)
    for i in trainUserRDD[data[0]]:
        sum1 += trainUserRDD[data[0]][i]
    active_user_avg = sum1 / len(trainUserRDD[data[0]])
    if data[1] not in businessMAP.keys():
        return ((data[0],data[1]),active_user_avg)
    c_pairs = []
    for i in businessMAP[data[1]]:
        c_pairs.append((data[0],i))
    c_w = []
    for iterator in c_pairs:
        value1 = trainUserRDD[iterator[0]]
        value2 = trainUserRDD[iterator[1]]
        setvalue1 = set(value1.keys())
        setvalue2 = set(value2.keys())
        corated_business1 = []
        corated_business2 = []
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        sum_num1 = 0
        sum_num2 = 0
        intersect = setvalue1.intersection(setvalue2)
        len_intersect = len(intersect)
        # Mean can also be calculated using Python Mean Function
        if len_intersect > 1:
            for i in intersect:
                corated_business1.append(value1[i])
                corated_business2.append(value2[i])
                num1 = value1[i]
                num2 = value1[i]
                sum_num1 += num1
                sum_num2 += num2
            mean1 = sum_num1/len_intersect
            mean2 = sum_num2/len_intersect
            for i,j in zip(corated_business1,corated_business2):
                numerator += ((i-mean1)*(j-mean2))
                denominator1 += (i-mean1)**2
                denominator2 += (j-mean2)**2
            weight_denominator = math.sqrt(denominator1*denominator2)
            if denominator1 !=0 and denominator2 !=0:
                weight = numerator/ weight_denominator
            else:
                weight = 0
            c_w.append((iterator,weight))
        else:
            c_w.append((iterator,0))

    numerater = 0
    denominator = 0
    for i in c_w:
        if i[1] is not 0:
            dc = trainUserRDD[i[0][1]]
            p_sum = 0
            p_total = 0
            for k, v in dc.items():
                if k is not data[1]:
                    p_sum += v
                    p_total += 1
            m_fk = p_sum/p_total
            numerater += ((dc[data[1]] - m_fk) * i[1])
            denominator += abs(i[1])

    if denominator != 0:
        predictions1 = numerater / denominator
        predictions111 = predictions1 + active_user_avg
    else:
        predictions111 = active_user_avg
    if predictions111 >= 5:
        predictions111 = active_user_avg
    elif predictions111 < 1.5:
        predictions111 = active_user_avg

    return ((data[0],data[1]),predictions111)


def getPredictionItem(data):
    if data[0] not in ubMAP.keys():
        return ((data[0], data[1]), 3)

    if data[1] not in businessList:
        return ((data[0], data[1]), 3)

    c_pairs = []
    for i in userMAP[data[0]]:
        c_pairs.append((data[1], i))
    c_w = []

    for iterator in c_pairs:
        value1 = trainBusinessRDD[iterator[0]]
        value2 = trainBusinessRDD[iterator[1]]
        setvalue1 = set(value1.keys())
        setvalue2 = set(value2.keys())
        corated_business1 = []
        corated_business2 = []
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        intersect = setvalue1.intersection(setvalue2)
        len_intersect = len(intersect)
        if len_intersect > 35:
            total_sum1 = 0
            total_sum2 = 0
            for i, j in zip(value1, value2):
                total_sum1 += value1[i]
                total_sum2 += value2[j]
            total_avg1 = total_sum1 / len(value1)
            total_avg2 = total_sum2 / len(value2)
            for i in intersect:
                corated_business1.append(value1[i])
                corated_business2.append(value2[i])
            for i, j in zip(corated_business1, corated_business2):
                numerator += ((i - total_avg1) * (j - total_avg2))
                denominator1 += (i - total_avg1) ** 2
                denominator2 += (j - total_avg2) ** 2
            weight_denominator = math.sqrt(denominator1 * denominator2)
            if weight_denominator != 0:
                weight = numerator / weight_denominator
            else:
                weight = 0
            c_w.append((iterator, weight))
        else:
            c_w.append((iterator, 0))
    numerater = 0
    denominator = 0
    for i in c_w:
        if i[1] is not 0:
            u_business_rate = ubMAP[data[0]]
            numerater += ( u_business_rate[i[0][1]]* i[1])
            denominator += abs(i[1])
    sum1 = 0
    for i in trainBusinessRDD[data[1]]:
        sum1 += trainBusinessRDD[data[1]][i]
    active_user_avg = sum1 / len(trainBusinessRDD[data[1]])
    if denominator != 0:
        predictions1 = numerater / denominator
        predictions111 = predictions1
    else:
        predictions111 = active_user_avg
    if predictions111 > 5:
        predictions111 = active_user_avg
    elif predictions111 < 1:
        predictions111 = active_user_avg

    return ((data[0], data[1]), predictions111)



# Load and parse the data
traindata = sc.textFile(sys.argv[1])
testdata = sc.textFile(sys.argv[2])
op_file = sys.argv[4]

traindata_temp = traindata.map(lambda l: l.split(',')).filter(lambda x: x[0] != 'user_id')
testdata_temp1 = testdata.map(lambda l: l.split(',')).filter(lambda x: x[0] != 'user_id')
if int(sys.argv[3]) == 1:
    userlist = traindata_temp.map(lambda x:x[0]).collect()
    userlisteval = testdata_temp1.map(lambda x:x[0]).collect()
    f_userlist = userlist + userlisteval

    businesslist = traindata_temp.map(lambda x:x[1]).collect()
    businesslisteval = testdata_temp1.map(lambda x:x[1]).collect()
    f_businesslist = businesslist + businesslisteval
    u_id ={}
    b_id={}
    u_count = 1
    b_count = 1
    res_u = {}
    res_b = {}

    for k in f_userlist:
        if k not in u_id.keys():
            u_id[k] = u_count
            u_count +=1
    for k in f_businesslist:
        if k not in b_id.keys():
            b_id[k] = b_count
            b_count += 1

    res_u = dict((v, k) for k, v in u_id.items())
    res_b = dict((v, k) for k, v in b_id.items())

    ratings = traindata_temp.map(lambda l: Rating(int(u_id[l[0]]), int(b_id[l[1]]), float(l[2])))

    # Build the recommendation model using Alternating Least Squares 
    rank = 2
    numIterations = 17
    t = 0.1
    model = ALS.train(ratings, rank, numIterations,t)

    # Evaluate the model on training data
    original = testdata_temp1.map(lambda x:((x[0],x[1]),float(x[2])))
    rating1 = testdata_temp1.map(lambda l: Rating(int(u_id[l[0]]),int(b_id[l[1]]),float(l[2])))
    testdata = rating1.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    printRDD1 = predictions.map(lambda x:((res_u[x[0][0]],res_b[x[0][1]]),x[1]))
    coldStart = testdata_temp1.map(lambda x:((x[0],x[1]),x[2]))
    differenceRDD = coldStart.subtractByKey(printRDD1)
    differenceRDD = differenceRDD.map(lambda x:((x[0]),3.0))
    finalrdd = printRDD1.union(differenceRDD)
    file = open(op_file,"w")
    file.write("user_id, business_id, prediction\n")
    for k,v in printRDD1.collectAsMap().items():
        file.write(str(k[0])+","+str(k[1])+","+str(v)+"\n")
    for k,v in differenceRDD.collectAsMap().items():
        file.write(str(k[0])+","+str(k[1])+","+str(3.0)+"\n")
    file.close()
    ratesAndPreds = finalrdd.map(lambda r: ((r[0][0], r[0][1]), r[1])).join(original).map(lambda tup: tup[1])
    metrics = RegressionMetrics(ratesAndPreds)
    print("RMSE = %s" % metrics.rootMeanSquaredError)
    print("Duration: ",time.time()-st)

elif int(sys.argv[3]) == 2:
    trainUserRDD = traindata_temp.map(lambda x: (x[0],(x[1],float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
    #format (userid,(dict(business_id,star))
    businessMAP = traindata_temp.map(lambda x: (x[1],x[0])).groupByKey().collectAsMap()
    pre_Result = testdata_temp1.map(getprediction)
    file = open(op_file,"w")
    stringWrite = "user_id, business_id, prediction\n"
    for i in pre_Result.collect():
        stringWrite += (str(i[0][0])+","+str(i[0][1])+","+str(i[1])+"\n")
    file.write(stringWrite)
    file.close()
    # Check RMSE Value
    # testdata = testdata_temp1.map(lambda x:((x[0],x[1]),float(x[2])))
    # joinRDD = pre_Result.join(testdata).map(lambda x:(x[0],abs(x[1][0]-x[1][1])))
    # rmse_numerator = joinRDD.map(lambda x:x[1]**2).reduce(add)
    # rmse = math.sqrt(rmse_numerator/testdata_temp1.count())
    #
    # print("RMSE:",rmse)
    print("Duration: ",time.time()-st)

elif int(sys.argv[3]) == 3:
    trainBusinessRDD = traindata_temp.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
    userMAP = traindata_temp.map(lambda x: (x[0], x[1])).groupByKey().collectAsMap()
    ubMAP = traindata_temp.map(lambda x: (x[0], (x[1],float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
    businessList = traindata_temp.map(lambda x:x[1]).collect()
    pre_Result = testdata_temp1.map(getPredictionItem)
    dictPrediction = pre_Result.collectAsMap()

    file = open(op_file,"w")
    stringWrite = "user_id, business_id, prediction\n"
    for k, v in dictPrediction.items():
        stringWrite += (str(k[0])+ ","+str(k[1])+","+ str(v) + "\n")
    file.write(stringWrite)
    file.close()
    # Check RMSE Value
    # testdata = testdata_temp1.map(lambda x: ((x[0], x[1]), float(x[2])))
    # joinRDD = pre_Result.join(testdata).map(lambda x: (x[0], abs(x[1][0] - x[1][1])))
    # rmse_numerator = joinRDD.map(lambda x: x[1] ** 2).reduce(add)
    # rmse = math.sqrt(rmse_numerator / testdata_temp1.count())
    # print("RMSE:", rmse)
    print("Duration: ", time.time() - st)

else:
    print("Invalid")
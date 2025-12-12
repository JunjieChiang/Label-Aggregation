package ceka.WSLC.code;

import ceka.core.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.HashMap;

public class wslc {
    //Number of class
    public static int m_numClass;
    //Worker similarity matrix
    public static double[][] maxtri;
    //Number of similar workers
    public static int numK;
    //Number of workers
    public static int numWorker;

    //Label completion
    public Dataset doInference(Dataset dataset) throws Exception {

        Dataset tempDataset = copyDataset(dataset);
        m_numClass = dataset.numClasses();
        numWorker = dataset.getWorkerSize();
        // Construct a dataset for each worker
        HashMap<String,Dataset> sub_dataset = new HashMap<String, Dataset>();
        for(int i = 0; i < tempDataset.getExampleSize(); i++) {
            Example example = tempDataset.getExampleByIndex(i);
            ArrayList<String> workerList = example.getWorkerIdList();
            for(int j = 0; j < workerList.size(); j++) {
                String id = workerList.get(j);
                if(!sub_dataset.containsKey(id)) {
                    Dataset temp_dataset = new Dataset(tempDataset, 0);
                    sub_dataset.put(id, temp_dataset);
                }
                sub_dataset.get(id).addExample(example);
            }
        }
        numK = numWorker;

        // Calculate feature vectors for each worker
        HashMap<String, double[]> attributeW = new HashMap<String, double[]>();
        for (String id : sub_dataset.keySet()) {
            double[] attW = new double[tempDataset.numAttributes()];
            attW = calAttributeW(sub_dataset.get(id), id);
            attributeW.put(id, attW);
        }

        //Calculating worker similarity
        maxtri = new double[tempDataset.getWorkerSize()][tempDataset.getWorkerSize()];

        //Worker index corresponding to similar workers for each worker
        int[][] similarKWorker = new int[tempDataset.getWorkerSize()][numK];

        //Obtain the worker weights according to the worker similarity
        double[][] distanceWorker = new double[tempDataset.getWorkerSize()][numK];

        for (int i = 0; i < tempDataset.getWorkerSize(); i++) {
            for (int j = 0; j < tempDataset.getWorkerSize(); j++) {
                String id1 = tempDataset.getWorkerByIndex(i).getId();
                String id2 = tempDataset.getWorkerByIndex(j).getId();
                if (i == j) {
                    maxtri[i][j] = 0.0;
                }
                else {
                    maxtri[i][j] = calSimilarity(attributeW.get(id1), attributeW.get(id2));
                }
            }
            double[] findK = maxtri[i];
            for (int k = 0; k < numK; k++) {
                similarKWorker[i][k] = Utils.maxIndex(findK);
                distanceWorker[i][k] = findK[Utils.maxIndex(findK)];
                findK[Utils.maxIndex(findK)] = Double.MIN_VALUE;
            }
        }

        //Label completion
        for (int i = 0; i < tempDataset.getExampleSize(); i++) {
            Example example = dataset.getExampleByIndex(i);
            Example e = tempDataset.getExampleByIndex(i);
            ArrayList<String> worker = example.getWorkerIdList();
            for (int j = 0; j < tempDataset.getWorkerSize(); j++) {
                String id = tempDataset.getWorkerByIndex(j).getId();
                if (!worker.contains(id)) {
                    int pre = getLabel(similarKWorker[j], distanceWorker[j], example, tempDataset);
                    Label label = new Label(null, new Integer(pre).toString(), example.getId(), id);
                    e.addNoisyLabel(label);
                }
            }
        }
        return tempDataset;
    }

    public static int getLabel(int[] kWorker, double[] distanceWorker, Example example, Dataset tempdata) {
        double[] pre = new double[m_numClass];
        ArrayList<String> worker = example.getWorkerIdList();
        //Find similar workers' labels
        double[] klabel = new double[numK];
        MultiNoisyLabelSet mnls = example.getMultipleNoisyLabelSet(0);
        for (int i = 0; i < kWorker.length; i++) {
            String id = tempdata.getWorkerByIndex(kWorker[i]).getId();
            if (!worker.contains(id)) {
                continue;
            }
            klabel[i] = mnls.getNoisyLabelByWorkerId(id).getValue();
            pre[(int) klabel[i]] += distanceWorker[i];
        }
        return Utils.maxIndex(pre);
    }

    //Calculating worker similarity (cosine similarity)
    public static double calSimilarity(double[] w1, double[] w2) {
        double s = 0;
        double fenzi = 0;
        double fenmu1 = 0;
        double fenmu2 = 0;
        for (int i = 0; i < w1.length; i++) {
            fenzi += w1[i] * w2[i];
            fenmu1 += w1[i] * w1[i];
            fenmu2 += w2[i] * w2[i];
        }
        if(fenzi==0||fenmu1==0||fenmu2==0){
            return 0;
        } else {
            s = fenzi / (Math.sqrt(fenmu1) * Math.sqrt(fenmu2));
            return 0.5 * (1 + s);
        }
    }

    //计算工人特征向量
    public static double[] calAttributeW(Dataset data, String id) throws Exception {
        Instances instances = new Instances(data);
        instances.setClassIndex(instances.numAttributes()-1);
        Dataset dataset = copyDataset(data);
        int numDataset = instances.numInstances();
        int numClass = instances.numClasses();
        //worker's feature vector
        double[] r = new double[instances.numAttributes()-1];
        //class probability distribution
        double[] pc = new double[numClass];
        double[][] classy = new double[numClass][numDataset];
        for (int j = 0; j < instances.numInstances(); j++) {
            Example example = dataset.getExampleByIndex(j);
            Instance instance = instances.instance(j);
            double Class = example.getNoisyLabelByWorkerId(id).getValue();
            instance.setClassValue(Class);
            pc[(int)Class]++;
            classy[(int)Class][j] = 1;
        }
        myNormalize(pc, numDataset);
        for (int i = 0; i < instances.numAttributes()-1; i++) {
            int numAtt = instances.attribute(i).numValues();
            double[][] attx = new double[numAtt][numDataset];
            double[] attx2 = new double[numDataset];
            double[][] paic = new double[numAtt][numClass];
            if (instances.attribute(i).isNominal()) {
                for (int j = 0; j < numDataset; j++) {
                    Instance instance = instances.instance(j);
                    paic[(int) instance.value(i)][(int) instance.classValue()]++;
                    attx[(int) instance.value(i)][j]=1;
                }
            } else {
                for (int j = 0; j < numDataset; j++) {
                    attx2[j] = instances.instance(j).value(i);
                }
            }
            //Calculate the correlation coefficient
            if (instances.attribute(i).isNominal()) {
                for (int q = 0; q < numClass; q++) {
                    for (int p = 0; p < numAtt; p++) {
                        paic[p][q] /= numDataset;
                        if (paic[p][q] != 0) {
                            r[i] += paic[p][q] * calAttributeR(attx[p], classy[q]);
                        }
                    }
                }
            } else {
                for (int q = 0; q < numClass; q++) {
                    if(pc[q]!=0) {
                        r[i] += pc[q] * calAttributeR(attx2, classy[q]);
                    }
                }
            }
        }
        return r;
    }

    //Calculate Pearson’s correlation
    public static double calAttributeR(double[] x, double[] y) throws Exception {
        double r = 0;
        double meanx = Utils.mean(x);
        double meany = Utils.mean(y);
        double fenzi = 0;
        double fenmu1 = 0;
        double fenmu2 = 0;
        for(int i = 0; i < x.length; i++) {
            fenzi += (x[i] - meanx)*(y[i] - meany);
            fenmu1 += Math.pow(x[i] - meanx,2);
            fenmu2 += Math.pow(y[i] - meany,2);
        }
        r = fenzi / (Math.sqrt(fenmu1) * Math.sqrt(fenmu2));
        if(Double.isNaN(r)) {
            r=0;
        }
        return r;
    }

    //Transformation into a probability distribution
    public static double[] myNormalize(double[] doubles, int numInstances) {
        for(int i = 0; i < doubles.length; ++i) {
            doubles[i] = doubles[i] / numInstances;
        }
        return doubles;
    }

    public static Dataset copyDataset(Dataset dataset) {
        Dataset copyDataset = new Dataset(dataset, 0);
        for (int k = 0; k < dataset.getExampleSize(); k++) {
            Example example = dataset.getExampleByIndex(k);
            copyDataset.addExample(example);
        }
        for (int k = 0; k < dataset.getCategorySize(); k++) {
            Category category = dataset.getCategory(k);
            copyDataset.addCategory(category);
        }
        for (int k = 0; k < dataset.getWorkerSize(); k++) {
            Worker worker = dataset.getWorkerByIndex(k);
            copyDataset.addWorker(worker);
        }
        return copyDataset;
    }
}

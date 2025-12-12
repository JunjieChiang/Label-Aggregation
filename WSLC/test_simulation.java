package ceka.WSLC.code;

import ceka.consensus.MajorityVote;
import ceka.consensus.gtic.GTIC;
import ceka.converters.FileLoader;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.others.DEWSMV.OptimizedError;
import ceka.others.MNLDP.MNLDP;
import ceka.simulation.MockWorker;
import ceka.utils.CekaUtils;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


public class test_simulation {
    private static Dataset m_dataset = null;
    private static Dataset m_dataset2 = null;
    private static HashMap<Integer, List> workerLabelList;
    private static String[] dataname= {"anneal","audiology","autos","balance-scale","biodeg","breast-cancer",
            "breast-w", "car", "credit-a","credit-g", "diabetes", "heart-c", "heart-h", "heart-statlog",
            "hepatitis", "horse-colic", "hypothyroid","ionosphere", "iris", "kr-vs-kp", "labor",
            "letter","lymph", "mushroom", "segment", "sick","sonar", "spambase","tic-tac-toe",
            "vehicle", "vote","vowel", "waveform", "zoo"
            };

    public void readData(int m_choose, double quality, int numWorker) throws Exception {
        String arffPath = "dataset/simulation/" + dataname[m_choose] + ".arff";
        m_dataset = FileLoader.loadFile(arffPath);
        //Handling of missing values
        ReplaceMissingValues m_Missing = new ReplaceMissingValues();
        m_Missing.setInputFormat(m_dataset);
        Instances instances = Filter.useFilter(m_dataset, m_Missing);
        m_dataset = CekaUtils.instancesToDataset(instances,m_dataset);
        mysimulateDataset(m_dataset, quality, numWorker);
    }

    public static Dataset Datasetcopy(Dataset dataset) {
        Dataset newdataset = dataset.generateEmpty();
        int numCateSize = dataset.getCategorySize();
        for (int i = 0; i < numCateSize; i++) {
            Category cate = dataset.getCategory(i);
            newdataset.addCategory(cate.copy());
        }
        for (int j = 0; j < dataset.getExampleSize(); j++) {
            Example example = dataset.getExampleByIndex(j);
            newdataset.addExample(example);
        }
        for (int i = 0; i < dataset.getWorkerSize(); i++) {
            newdataset.addWorker(dataset.getWorkerByIndex(i));
        }
        return newdataset;
    }

    public double integrationAccuracy(Dataset dataset) {
        int numCleanExample = 0;
        for (int i = 0; i < dataset.getExampleSize(); i++) {
            Example example = dataset.getExampleByIndex(i);
            if (example.getIntegratedLabel().getValue() == example.getTrueLabel().getValue())
                numCleanExample++;
        }
        return numCleanExample / (double) dataset.getExampleSize();
    }

    public static void mysimulateDataset(Dataset dataset, double quality, int numWorkers) {
        MockWorker[] mockWorkers = new MockWorker[numWorkers];
        for (int id = 0; id < numWorkers; id++) {
            mockWorkers[id] = new MockWorker(String.valueOf(id));
        }
        //Uniform distribution
        LabelStrategy strategy = new LabelStrategy(quality);
        strategy.RandomAssignWorkerQuality(mockWorkers, 0.5, 0.4);

        //Gaussian distribution
//        GaussianLabelingStrategy strategy = new GaussianLabelingStrategy(0.7,0.15);
//        strategy.assignWorkerQuality(mockWorkers);

        workerLabelList = new HashMap<Integer, List>();
        double[] workerP = new double[numWorkers];
        for (int id = 0; id < numWorkers; id++) {
            List instList = new ArrayList();
            workerLabelList.put(id, instList);
            workerP[id] = Math.random() * 0.4 + 0.1;
            System.out.println(workerP[id]);
        }
        for(int i = 0; i < dataset.getExampleSize(); i++) {
            int flag = 0;
            for (int j = 0; j < numWorkers; j++) {
                if (Math.random() < workerP[j]) {
                    workerLabelList.get(j).add(i);
                    flag += 1;
                }
            }
            if (flag == 0) {
                i -= 1;
            }
        }

        for (int j = 0; j < numWorkers; j++) {
            mockWorkers[j].labelingMy(dataset, strategy,workerLabelList.get(j));
        }

    }

    public static void main(String[] args) {

        double meanIntegrationMV = 0;
        double meanIntegrationGTIC = 0.0;
        double meanIntegrationDEWSMV = 0.0;
        double meanIntegrationMNLDP = 0.0;

        double meanIntegrationMV2 = 0;
        double meanIntegrationGTIC2 = 0.0;
        double meanIntegrationDEWSMV2 = 0.0;
        double meanIntegrationMNLDP2 = 0.0;

        double quality = 0.55;
        int numWorker = 40;

        try {
            String resultPath1 = "test_simulation.txt";
            FileOutputStream fs1 = new FileOutputStream(new File(resultPath1));
            PrintStream result1 = new PrintStream(fs1);
            result1.format("%-20s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s",
                    "Dataset", "MV", "MV-WSLC", "GTIC", "GTIC-WSLC", "DEWSMV", "DEWSMV-WSLC", "MNLDP", "MNLDP-WSLC");
            result1.println();

            for (int i = 0; i < dataname.length; i++) {
                double IntegrationMV = 0;
                double IntegrationGTIC = 0;
                double IntegrationDEWSMV = 0;
                double IntegrationMNLDP = 0;

                double IntegrationMV2 = 0;
                double IntegrationGTIC2 = 0;
                double IntegrationDEWSMV2 = 0;
                double IntegrationMNLDP2 = 0;

                double times = 10;
                for (int m = 1 ; m <= times; m++) {
                    test_simulation expriment = new test_simulation();
                    Boolean flag = true;
                    while (flag) {
                        expriment.readData(i, quality, numWorker);
                        Boolean flagWorker = false;
                        for (int workerIndex : workerLabelList.keySet()) {
                            int num = workerLabelList.get(workerIndex).size();
                            if (num == 0) {
                                flagWorker = true;
                                break;
                            }
                        }
                        if (flagWorker) {
                            flag = true;
                        }else {
                            flag = false;
                        }
                    }

                    //Label completion
                    Dataset temp = CekaUtils.datasetCopy(m_dataset);
                    wslc similarity = new wslc();
                    m_dataset2 = similarity.doInference(temp);

                    // Majority Voting
                    Dataset tempDataMV = CekaUtils.datasetCopy(m_dataset);
                    MajorityVote mv = new MajorityVote();
                    mv.doInference(tempDataMV);
                    IntegrationMV += expriment.integrationAccuracy(tempDataMV);

                    Dataset tempDataMV2 = CekaUtils.datasetCopy(m_dataset2);
                    MajorityVote mv2 = new MajorityVote();
                    mv2.doInference(tempDataMV2);
                    IntegrationMV2 += expriment.integrationAccuracy(tempDataMV2);

                    // GTIC
                    Dataset datasetGTIC = CekaUtils.datasetCopy(m_dataset);
                    GTIC gtic = new GTIC("GTIC/");
                    gtic.doInference(datasetGTIC);
                    IntegrationGTIC += expriment.integrationAccuracy(datasetGTIC);

                    Dataset datasetGTIC2 = CekaUtils.datasetCopy(m_dataset2);
                    GTIC gtic2 = new GTIC("GTIC/");
                    gtic2.doInference(datasetGTIC2);
                    IntegrationGTIC2 += expriment.integrationAccuracy(datasetGTIC2);

                    //DEWSMV
                    Dataset tempDataDEWSMV = Datasetcopy(m_dataset);
                    OptimizedError dewsmv = new OptimizedError();
                    dewsmv.DE_search(tempDataDEWSMV);
                    IntegrationDEWSMV += expriment.integrationAccuracy(tempDataDEWSMV);

                    Dataset tempDataDEWSMV2 = Datasetcopy(m_dataset2);
                    OptimizedError dewsmv2 = new OptimizedError();
                    dewsmv2.DE_search(tempDataDEWSMV2);
                    IntegrationDEWSMV2 += expriment.integrationAccuracy(tempDataDEWSMV2);

                    //MNLDP
                    Dataset tempDataMNLDP = Datasetcopy(m_dataset);
                    MNLDP mnldp = new MNLDP();
                    mnldp.doInference(tempDataMNLDP);
                    IntegrationMNLDP += expriment.integrationAccuracy(tempDataMNLDP);

                    Dataset tempDataMNLDP2 = Datasetcopy(m_dataset2);
                    MNLDP mnldp2 = new MNLDP();
                    mnldp2.doInference(tempDataMNLDP2);
                    IntegrationMNLDP2 += expriment.integrationAccuracy(tempDataMNLDP2);
                }
                IntegrationMV /= times;
                IntegrationGTIC /= times;
                IntegrationDEWSMV /= times;
                IntegrationMNLDP /= times;

                IntegrationMV2 /= times;
                IntegrationGTIC2 /= times;
                IntegrationDEWSMV2 /= times;
                IntegrationMNLDP2 /= times;

                meanIntegrationMV += IntegrationMV;
                meanIntegrationGTIC += IntegrationGTIC;;
                meanIntegrationDEWSMV += IntegrationDEWSMV;
                meanIntegrationMNLDP += IntegrationMNLDP;

                meanIntegrationMV2 += IntegrationMV2;
                meanIntegrationGTIC2 += IntegrationGTIC2;
                meanIntegrationDEWSMV2 += IntegrationDEWSMV2;
                meanIntegrationMNLDP2 += IntegrationMNLDP2;

                result1.format("%-20s %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f", dataname[i],
                        IntegrationMV, IntegrationMV2, IntegrationGTIC, IntegrationGTIC2,
                        IntegrationDEWSMV, IntegrationDEWSMV2, IntegrationMNLDP, IntegrationMNLDP2);
                result1.println();
            }

            meanIntegrationMV /= dataname.length;
            meanIntegrationGTIC /= dataname.length;
            meanIntegrationDEWSMV /= dataname.length;
            meanIntegrationMNLDP /= dataname.length;

            meanIntegrationMV2 /= dataname.length;
            meanIntegrationGTIC2 /= dataname.length;
            meanIntegrationDEWSMV2 /= dataname.length;
            meanIntegrationMNLDP2 /= dataname.length;

            result1.format("%-20s %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f","Mean",
                    meanIntegrationMV, meanIntegrationMV2, meanIntegrationGTIC, meanIntegrationGTIC2,
                    meanIntegrationDEWSMV, meanIntegrationDEWSMV2, meanIntegrationMNLDP,meanIntegrationMNLDP2);
            result1.println();
            result1.close();
        } catch (Exception e) {
            System.err.println(e.getMessage());        }
    }
}




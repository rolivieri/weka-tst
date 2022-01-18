package ml.olivieri;

import java.util.ArrayList;
import java.io.*;
import weka.classifiers.*;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.*;

public class MyClassifier {

    public static Evaluation getClassification(Instances trainData, Instances testData, Classifier classifierModel)
            throws Exception {

        // Build model
        classifierModel.buildClassifier(trainData);
        // Classify test data
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifierModel, testData);
        return eval;
    }

    public static Instances createInstances(String fileName) throws Exception {
        // Read data
        Reader targetReader = null;
        try {
            InputStream inStream = MyClassifier.class.getClassLoader().getResourceAsStream(fileName);
            targetReader = new InputStreamReader(inStream);
            Instances dataSet = new Instances(targetReader);
            int noAttributes = dataSet.numAttributes() - 1;
            // Set index for class attribute
            dataSet.setClassIndex(noAttributes);
            return dataSet;
        } finally {
            if (targetReader != null) {
                targetReader.close();
            }
        }
    }

    public static Classifier[] createClassifiers() {
        J48 model1 = new J48();
        DecisionTable model2 = new DecisionTable();
        DecisionStump model3 = new DecisionStump();
        PART model4 = new PART();
        Classifier[] classifierModel = new Classifier[4];
        classifierModel[0] = model1;
        classifierModel[1] = model2;
        classifierModel[2] = model3;
        classifierModel[3] = model4;
        return classifierModel;
    }

    public static void main(String[] args) throws Exception {
        // Create instances
        Instances inputData = createInstances("weather_details.arff");
        System.out.println(inputData.toSummaryString() + '\n');

        // Splits for 10-fold cross validation
        int numFolds = 10;
        Instances[] trainData = new Instances[numFolds];
        Instances[] testData = new Instances[numFolds];
        for (int numFold = 0; numFold < numFolds; numFold++) {
            // train
            trainData[numFold] = inputData.trainCV(numFolds, numFold);
            // test
            testData[numFold] = inputData.testCV(numFolds, numFold);
        }

        // Create classifiers
        Classifier[] classifiers = createClassifiers();

        for (Classifier classifier : classifiers) {
            ArrayList<Prediction> results = new ArrayList<Prediction>();
            for (int s = 0; s < trainData.length; s++) {
                Evaluation eval = getClassification(trainData[s], testData[s], classifier);
                // append to result
                results.addAll(eval.predictions());
            }

            // Model Name
            String modelName = classifier.getClass().getSimpleName();
            System.out.println("************************************");
            System.out.println("Model Name : " + modelName);
            // Accuracy
            int matchCount = 0;
            for (Prediction prediction : results) {
                if (prediction.predicted() == prediction.actual())
                    matchCount++;
            }

            double accuracyPercentage = ((double) matchCount / results.size()) * 100;
            System.out.println(String.format("Accuracy is : %,.2f", accuracyPercentage));
            System.out.println("************************************\n");
        }
    }
}

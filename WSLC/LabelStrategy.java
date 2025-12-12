package ceka.WSLC.code;

import ceka.simulation.MockWorker;
import ceka.simulation.SingleQualLabelingStrategy;

import java.util.Random;

public class LabelStrategy extends SingleQualLabelingStrategy {

	public LabelStrategy(double p) {
		super(p);
		// TODO Auto-generated constructor stub
	}

	public void RandomAssignWorkerQuality(MockWorker[] workers,double baseline, double var) {
		for (int i = 0; i < workers.length; i++) {
			double pro = 0;
			Random random = new Random();
			pro = random.nextDouble() * var + baseline;
			workers[i].setSingleQuality(pro);
		}
	}

}

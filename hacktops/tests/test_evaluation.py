from unittest import TestCase
import pandas as pa
import numpy as np

from hacktops.evaluate import recall_tops
from hacktops.predict import predict

class TestEvaluation(TestCase):

    def setUp(self):
        self.df_logs = pa.DataFrame(columns = ["wellName","Depth","GR"] )

        nb_depth = 20
        self.nb_wells = 3

        index_a = 5
        index_b = 15

        depths = np.hstack([np.arange(nb_depth)]*self.nb_wells)
        wells = [ ["well_{0}".format(i)]*nb_depth for i in range(self.nb_wells)]
        gr_values = [ 1 for i in range(nb_depth) ]
        gr_values[index_a] = 2
        gr_values[index_b] = 3


        self.df_logs["wellName"] = np.hstack(wells)
        self.df_logs["Depth"] = depths
        self.df_logs["GR"] = np.hstack([gr_values]*self.nb_wells)

        self.tops = ["A","B"]

        self.df_tops_true = pa.DataFrame( data = np.reshape([index_a,index_b]*3,[3,2]), index = ["well_{0}".format(i) for i in range(self.nb_wells) ],columns = self.tops)


    def test_predict(self):
        df_tops_pred = predict(self.df_logs,tops= self.tops)  
        wells = ["well_{0}".format(i) for i in range(self.nb_wells)]
        self.assertListEqual(list(df_tops_pred.columns), self.tops)
        self.assertListEqual(list(df_tops_pred.index), wells)

    def test_recall_one(self):
        df_tops_pred = predict(self.df_logs,tops= self.tops)  
        df_tops_pred[self.tops[0]] = 5
        df_tops_pred[self.tops[1]] = 15
        recall, wrmse,df = recall_tops(self.df_tops_true,df_tops_pred)        
        self.assertEqual(recall, 1)
        self.assertEqual(wrmse,0)

    def test_recall_half(self):
        df_tops_pred = predict(self.df_logs,tops= self.tops)  
        df_tops_pred[self.tops[0]] = 3
        df_tops_pred[self.tops[1]] = 15
        recall, wrmae,df = recall_tops(self.df_tops_true,df_tops_pred,tolerance=1)        
        self.assertEqual(recall, 0.5)
        self.assertEqual(wrmae,1)

    def test_recall_missing(self):
        df_tops_pred = predict(self.df_logs,tops= self.tops)  
        df_tops_pred[self.tops[0]] = 5
        df_tops_pred[self.tops[1]] = 15
        df_tops_pred = df_tops_pred.drop("well_1")                
        recall, wrmae,df = recall_tops(self.df_tops_true,df_tops_pred,tolerance=1)        
        self.assertEqual(recall, 4/6)
        self.assertEqual(wrmae,0)

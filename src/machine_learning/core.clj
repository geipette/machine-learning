(ns machine-learning.core
  (:require [machine-learning.network :refer :all]
            [machine-learning.mnist-reader :refer [load-testing-data load-validation-data load-training-data]])
  (:gen-class))

(defn do-and-print [f str]
  (println str)
  (f))

(defn -main [& args]
  (println "Loading data...")
  (let [testing_data (do-and-print load-testing-data "  loading testing data")
        validation_data (do-and-print load-validation-data "  loading validation data")
        training_data (do-and-print load-training-data "  loading trainig data")
        network (do-and-print #(create-network [784, 30, 10]) "  creating network")]
    (sgd network training_data 30 10 100.0 testing_data)))

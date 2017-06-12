(ns machine-learning.core
  (:require [machine-learning.network :refer :all]
            [machine-learning.mnist-reader :refer [load-testing-data load-validation-data load-training-data]]
            [clojure.tools.cli :refer [parse-opts]])
  (:gen-class))

(def cli-options
  ;; An option with a required argument
  [["-l" "--learn-rate LEARN_RATE" "The learn rate float"
    :default 3.0
    :parse-fn #(Double/parseDouble %)]
   ])

(defn do-and-print [f str]
  (println str)
  (f))

(defn sgd-mnist [learn_rate]
  (println "Running training with learn_rate: " learn_rate)
  (println "Loading data...")
  (let [testing_data (do-and-print load-testing-data "  loading testing data")
        validation_data (do-and-print load-validation-data "  loading validation data")
        training_data (do-and-print load-training-data "  loading trainig data")
        network (do-and-print #(create-network [784, 30, 10]) "  creating network")]
    (do-and-print #(sgd network training_data 30 10 learn_rate testing_data) "Running network"))
  )

(defn -main [& args]
  (let [options (parse-opts args cli-options)]
    (if-not (empty? (:errors options))
      (map println (:errors options))
      (sgd-mnist (-> options :options :learn-rate)))))

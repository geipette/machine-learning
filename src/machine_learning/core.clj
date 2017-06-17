(ns machine-learning.core
  (:require [machine-learning.network :refer :all]
            [machine-learning.mnist-reader :refer [load-testing-data load-validation-data load-training-data]]
            [clojure.tools.cli :as cli]
            [metrics.reporters.console :as console]
            [metrics.core :refer [default-registry]]
            [clojure.string :as str])
  (:gen-class))

(defn parse-int [str]
  (Integer/parseInt str))

(defn parse-double [str]
  (Double/parseDouble str))

(def cli-options
  ;; An option with a required argument
  [["-l" "--learn_rate LEARN_RATE" "The learn rate float"
    :default 0.5
    :parse-fn parse-double]
   ["-e" "--epocs EPOCS" "The number of epocs to train"
    :default 30
    :parse-fn parse-int]
   ["-bs" "--batch_size BATCH_SIZE" "The size of the training batches"
    :default 10
    :parse-fn parse-int]
   ["-c" "--cost COST" "The cost strategy to use. [cross-entropy|quadratic]"
    :default (->CrossEntropyCost)
    :default-desc "cross-entropy"
    :parse-fn #(case %
                 "cross-entropy" (->CrossEntropyCost)
                 "quadratic" (->QuadraticCost)
                 nil)
    :validate [#(not (nil? %)) "must be 'cross-entropy' or 'quadratic'"]]
   ["-d" "--lambda LAMBDA" "Regularization parameter"
    :default 5.0
    :parse-fn parse-double]
   ["-n" "--hidden HIDDEN" "The number of neurons in the hidden layer"
    :default 30
    :parse-fn parse-int]
   ["-h" "--help" "Print this help"]])

(defn usage [options-summary]
  (->> [""
        "Usage: mnist-trainer [options] action"
        ""
        "Options:"
        options-summary
        ""
        "Actions:"
        "  train    Train a network"
        ""]
       (str/join \newline)))

(defn error-msg [errors]
  (str/join \newline errors))

(defn validate-args
  "Validate command line arguments. Either return a map indicating the program
  should exit (with a error message, and optional ok status), or a map
  indicating the action the program should take and the options provided."
  [args]
  (let [{:keys [options arguments errors summary]} (cli/parse-opts args cli-options)]
    (cond
      (:help options) {:exit-message (usage summary) :ok? true}
      errors {:exit-message (str (error-msg errors) "\n" (usage summary))}
      (and (= 1 (count arguments)) (#{"train"} (first arguments))) {:action (first arguments) :options options}
      :else {:exit-message (usage summary) :ok? true})))

(defn exit [status msg]
  (println msg)
  (System/exit status))

(defn do-and-print [f str]
  (println str)
  (f))

(defn sgd-mnist [options]
  (println "Running training with: " options)
  (println "Loading data...")
  (let [testing_data (do-and-print load-testing-data "  loading testing data")
        validation_data (do-and-print load-validation-data "  loading validation data")
        training_data (do-and-print load-training-data "  loading training data")
        network (do-and-print #(create-network [784, (:hidden options), 10]) "  creating network")
        training_spec (map->TrainingSpec options)]
    ;(console/start (console/reporter reg  {}) 10)
    (do-and-print #(sgd network training_spec training_data testing_data) "Running network")))

(defn -main [& args]
  (let [{:keys [action options exit-message ok?]} (validate-args args)]
    (if exit-message
      (exit (if ok? 0 1) exit-message)
      (case action
        "train" (sgd-mnist options)))))

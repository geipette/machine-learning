(ns machine-learning.core
  (:require [machine-learning.network :refer :all]
            [machine-learning.mnist-reader :refer [load-data]]
            [machine-learning.report.image :refer [save-image-png!]]
            [machine-learning.report.html :refer [report write-index-html! ->ReportSettings]]
            [clojure.tools.cli :as cli]
            [metrics.reporters.console :as console]
            [metrics.core :refer [default-registry]]
            [clojure.string :as str]
            [clojure.java.io :as io])
  (:gen-class))

(defn load-testing-data []
  (load-data (io/resource "testing_data.clj.nippy")))

(defn load-training-data []
  (load-data (io/resource "training_data.clj.nippy")))

(defn load-validation-data []
  (load-data (io/resource "validation_data.clj.nippy")))

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
   ["-b" "--batch_size BATCH_SIZE" "The size of the training batches"
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
   ["-p" "--progress" "Report training progress"
    :default false]
   ["-s" "--save" "Save network"
    :default false]
   ["-f" "--file FILENAME" "The filename to save and/or load network"
    :default "target/last_network.clj.nippy"]
   ["-r" "--report DIR" "Makes a performance report based on the testing data in the supplied directory"]
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
        "  load     Load a network from file"
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
      (and (= 1 (count arguments)) (#{"train"
                                      "load"} (first arguments))) {:action (first arguments) :options options}
      :else {:exit-message (usage summary) :ok? true})))

(defn exit [status msg]
  (println msg)
  (System/exit status))

(defn do-and-print [f str]
  (println str)
  (f))

(defn sgd-info [options]
  (->> [" --- Training a new network ---"
        (format " Topology: [%s %s %s]" 784 (:hidden options) 10)
        (format " Learn rate: %s" (:learn_rate options))
        (format " L2 lambda: %s" (:lambda options))
        (format " Batch size: %s" (:batch_size options))
        (format " Epocs: %s" (:epocs options))
        (if (:save options)
          (format " Trained network will be saved as %s" (:file options))
          " Trained network will be discarded after use.")
        ]
       (str/join \newline)))

(defn sgd-mnist [options]
  (println (sgd-info options))
  (println "Loading data...")
  (let [testing_data (if (:progress options) (do-and-print load-testing-data "  loading testing data") nil)
        ;validation_data (do-and-print load-validation-data "  loading validation data")
        training_data (do-and-print load-training-data "  loading training data")
        network (do-and-print #(create-network [784, (:hidden options), 10]) "  creating network")
        training_spec (map->TrainingSpec options)]
    (println "Running network")
    (sgd network training_spec training_data testing_data)))

(defn acquire-network [action options]
  (case action
    "train" (sgd-mnist options)
    "load" (load-network (:file options))))

(defn -main [& args]
  (let [{:keys [action options exit-message ok?]} (validate-args args)]
    (if exit-message
      (exit (if ok? 0 1) exit-message)
      (let [network (acquire-network action options)]
        (if (:save options)
          (save-network network (:file options)))
        (if (:report options)
          (let [test_data (load-testing-data)]
            (println (format "%s / %s" (evaluate network test_data) (count test_data)))))
        (exit 0 "")))))

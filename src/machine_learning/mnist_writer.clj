(ns machine-learning.mnist-writer
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.core.matrix :as m :refer [reshape zero-array]]
            [taoensso.nippy :as nippy :refer [freeze-to-file]])
  (:import (java.util.zip GZIPOutputStream)))

(defn reshape-inputs [inputs]
  (mapv #(m/reshape % [784 1]) inputs))

(defn vectorized-result [digit]
  (assoc (m/zero-array [10 1]) digit [1.0]))

(defn save! [filename data]
  (nippy/freeze-to-file (io/file filename) data))

(defn read-json [filename]
  (with-open [reader (io/reader (io/file filename) :encoding "UTF-8")]
    (json/read reader)))

(defn process-resources []
  (println "Processing mnist_training.json...")
  (let [training_data (read-json "target/mnist_training.json")
        training_inputs (reshape-inputs (first training_data))
        training_results (mapv vectorized-result (second training_data))]
    (println "  writing training_data.clj.nippy...")
    (save! "target/training_data.clj.nippy" (mapv vector training_inputs training_results))
    (println "  done"))

  (println "Processing mnist_validation.json...")
  (let [validation_data (read-json "target/mnist_validation.json")
        validation_inputs (reshape-inputs (first validation_data))]
    (println "  writing validation_data.clj.nippy...")
    (save! "target/validation_data.clj.nippy" (mapv vector validation_inputs (second validation_data)))
    (println "  done"))

  (println "Processing mnist_testing.json...")
  (let [testing_data (read-json "target/mnist_testing.json")
        testing_inputs (reshape-inputs (first testing_data))]
    (println "  writing testing_data.clj.nippy...")
    (save! "target/testing_data.clj.nippy" (mapv vector testing_inputs (second testing_data)))
    (println "  done")))

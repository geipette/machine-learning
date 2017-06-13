(ns machine-learning.mnist-reader
  (:require [clojure.java.io :as io]
            [taoensso.nippy :as nippy :refer [freeze-to-file thaw-from-file]]))

(defn load-data [filename]
  (nippy/thaw-from-file (io/file filename)))

(defn load-testing-data []
  (load-data (io/resource "testing_data.clj.nippy")))

(defn load-training-data []
  (load-data (io/resource "training_data.clj.nippy")))

(defn load-validation-data []
  (load-data (io/resource "validation_data.clj.nippy")))
(ns machine-learning.mnist-reader
  (:require [clojure.java.io :as io]
            [taoensso.nippy :as nippy :refer [thaw-from-file thaw]])
  (:import (java.io InputStream ByteArrayInputStream DataInputStream BufferedInputStream)))

(defn load-data
  [^String filename]
  (nippy/thaw-from-file (io/file filename)))

(defn load-data! [^InputStream inputstream]
  (nippy/thaw-from-in! (DataInputStream. inputstream)))
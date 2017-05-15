(ns machine-learning.network
  (:require [clojure.core.matrix :as m]
            ;[uncomplicate.neanderthal.core :refer :all]
            ;[uncomplicate.neanderthal.native :refer :all]
            )
  (:import (java.util Random)))

(defn sample-gaussian
  ([n] (sample-gaussian n (Random.)))
  ([n rng]
   (into [] (repeatedly n #(.nextGaussian rng)))))

(defn- create-initial-weights [sizes]
  (mapv #(sample-gaussian (* (first %) (second %)))
        (map vector (butlast sizes) (rest sizes))))

(defn create-network [sizes]
  (if (= 0 (count sizes))
    (throw (IllegalArgumentException. (format "create-network requires sizes to have at least one element, was: %s" sizes))))
  {:num-layers (count sizes)
   :sizes sizes
   :biases (sample-gaussian (count (rest sizes)))
   :weights (create-initial-weights sizes)})

(defn sigmoid [z]
  (if-not (vector? z)
    (Math/exp z)
    (mapv sigmoid z)))

(defn- feedforward-layer-output [input layer]
  (sigmoid (mapv #(+ % (first layer)) (m/dot input (second layer)))))

(defn feed-forward [network input]
  "Return the output of the network for a given input"
  (loop [result input layers (map vector (:biases network) (:weights network))]
    (if (empty? layers)
      result
      (recur
        (feedforward-layer-output result (first layers))
        (rest layers)))))

(defn- sgd-epoc [network training_data mini_batch_size eta]
  (let [mini_batches (partition mini_batch_size (shuffle training_data))]
    ; TDOD doing nothing rigt now
    network))

;"Train the neural network using mini-batch stochastic
;  gradient descent.  The \"training_data\" is a list of tuples
;  \"(x, y)\" representing the training inputs and the desired
;  outputs.  The other non-optional parameters are
;  self-explanatory. If \"test_data\" is provided then the
;  network will be evaluated against the test data after each
;  epoch, and partial progress printed out.  This is useful for
;  tracking progress, but slows things down substantially."
(defn sgd
  ([network training_data epocs mini_batch_size eta]
    (sgd network training_data epocs mini_batch_size eta nil))
  ([network training_data epocs mini_batch_size eta test_data]
    (loop [epoc 0 result network]
      (if (> 0 epoc)
        (println (format "Epoch %s complete" epoc)))
      (if (>= epoc epocs)
        result
        (do
          (recur (inc epoc) (sgd-epoc result training_data mini_batch_size eta))))))
  )


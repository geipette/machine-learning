(ns machine-learning.network
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.random :as mr]
            [clojure.core.matrix.operators :refer [* + -]]

    ;[uncomplicate.neanderthal.core :refer :all]
    ;[uncomplicate.neanderthal.native :refer :all]
            )
  (:import (java.util Random)))

(m/set-current-implementation :vectorz)

(defn zero-array [array]
  (mapv m/zero-array (map m/shape array)))

(defn sample-gaussian
  ([n] (sample-gaussian n (Random.)))
  ([n rng]
   (if-not (coll? n)
     (into [] (repeatedly n #(.nextGaussian rng)))
     (mapv #(sample-gaussian % rng) n))))

(defn create-network [sizes]
  (if (>= 2 (count sizes))
    (throw (IllegalArgumentException. (format "create-network requires sizes to have at least one element, was: %s" sizes))))
  {:num-layers (count sizes)
   :sizes      sizes
   :biases     (map #(mr/sample-normal [% 1]) (rest sizes))
   :weights    (map #(mr/sample-normal [%1 %2]) (rest sizes) (butlast sizes))})

(defn sigmoid [z]
  (if-not (or (m/vec? z) (m/matrix? z))
    (/ 1.0 (+ 1 (Math/exp (- z))))
    (m/emap sigmoid z)))

(defn sigmoid-prime [z]
  (if-not (or (m/vec? z) (m/matrix? z))
    (* (sigmoid z) (- 1 (sigmoid z)))
    (let [sz (sigmoid z)]
      (* sz (mapv - (repeat 1) sz)))))

(defn assert-correct-input [input]
  (if (or (m/vec? input) (m/matrix? input))
    (let [shape (m/shape input)]
      (if (and (= 2 (count shape))
               (= 1 (second shape)))
        input))
    (throw (IllegalArgumentException. (format "input must be a vector with shape [n, 1], was %s" input)))))

(defn feed-forward [network input]
  "Return the output of the network for a given input"
  (loop [result (assert-correct-input input)
         biases (:biases network)
         weights (:weights network)]
    (if (empty? biases)
      result
      (let [output (sigmoid (+ (map #(m/inner-product % result) (first weights)) (first biases)))]
        (recur output (rest biases) (rest weights))))))

(defn cost-derivative [output_activations desired_output]
  (- output_activations desired_output))

(defn collect-activations [network input]
  (loop [activation (assert-correct-input input)
         biases (:biases network)
         weights (:weights network)
         activations [(m/matrix input)]
         zs []]
    (if (empty? biases)
      [activations zs]
      (let [z (+ (map #(m/inner-product % activation) (first weights)) (first biases))
            next_activation (sigmoid z)]
        (recur next_activation (rest biases) (rest weights) (conj activations (m/matrix next_activation)) (conj zs (m/matrix z)))))))

(defn backprop [network [input desired_output]]
  "Return a tuple '(nabla_b, nabla_w)' representing the
  gradient for the cost function C_x.  'nabla_b' and
  'nabla_w' are layer-by-layer lists of vectors."
  (let [[activations zs] (collect-activations network input)
        delta (* (cost-derivative (last activations) desired_output) (sigmoid-prime (last zs)))
        zero_b (zero-array (:biases network))
        zero_w (zero-array (:weights network))
        last_layer_index (- (:num-layers network) 2)]
    (loop [layers (range last_layer_index 0 -1)
           last_delta delta
           nabla_b (assoc zero_b last_layer_index delta)
           nabla_w (assoc zero_w last_layer_index (m/inner-product delta (m/transpose (nth activations last_layer_index))))]
      (if (empty? layers)
        [nabla_b nabla_w]
        (let [layer_index (first layers)
              sp (sigmoid-prime (nth zs (dec layer_index)))
              delta_b (* sp (m/inner-product (m/transpose (nth (:weights network) layer_index)) last_delta))
              delta_w (m/inner-product delta_b (m/transpose (nth activations (dec layer_index))))]
          (recur (rest layers) delta_b (assoc nabla_b (dec layer_index) delta_b) (assoc nabla_w (dec layer_index) delta_w)))))))

(defn accumulate-deltas [[acc_biases acc_weights] [delta_biases delta_weights]]
  [(map + acc_biases delta_biases) (map + acc_weights delta_weights)])

(defn backprop-batch [network batch]
  (reduce accumulate-deltas [(zero-array (:biases network)) (zero-array (:weights network))]
          (pmap #(backprop network %) batch)))

(defn apply-delta [original adjustment eta batch-size]
  (m/emap #(- %1 (* (/ eta batch-size) %2)) original adjustment))

(defn update-batch [network batch eta]
  "Update the network's weights and biases by applying
  gradient descent using backpropagation to a single mini batch.
  The \"batch\" is a list of tuples \"(x, y)\", and \"eta\"
  is the learning rate."
  (let [[delta_biases delta_weights] (backprop-batch network batch)
        batch-size (count batch)
        biases (map apply-delta (:biases network) delta_biases (repeat eta) (repeat batch-size))
        weights (map apply-delta (:weights network) delta_weights (repeat eta) (repeat batch-size))]
    (assoc network :weights weights :biases biases)))

(defn index-of-max [output]
  (first (apply max-key #(first (second %)) (map-indexed vector output))))

(defn run-test [network input]
  (index-of-max (feed-forward network input)))

(defn run-tests [network test_data]
  (loop [remaining_data test_data
         results []]
    (if (empty? remaining_data)
      results
      (let [input (ffirst remaining_data)
            output (run-test network input)
            expected-output (second (first remaining_data))]
        (recur (rest remaining_data) (conj results [output expected-output]))))))

(defn evaluate [network test_data]
  (let [test_results (run-tests network test_data)]
    (reduce (fn [result [x y]] (if (= x y) (inc result) result)) 0 test_results)))

(defn sgd-epoc [network training_data batch_size eta]
  (let [batches (partition batch_size batch_size nil (shuffle training_data))]
    (loop [updated_network network remaining_batches batches]
      (if (empty? remaining_batches)
        updated_network
        (recur (update-batch updated_network (first remaining_batches) eta) (rest remaining_batches))))))

(defn sgd
  ([network training_data epocs mini_batch_size eta]
   (sgd network training_data epocs mini_batch_size eta nil))
  ([network training_data epocs mini_batch_size eta test_data]
   "Train the neural network using mini-batch stochastic
     gradient descent.  The 'training_data' is a list of tuples
     '(x, y)' representing the training inputs and the desired
     outputs.  The other non-optional parameters are
     self-explanatory. If 'test_data' is provided then the
     network will be evaluated against the test data after each
     epoch, and partial progress printed out.  This is useful for
     tracking progress, but slows things down substantially."
   (loop [epoc 0 result network]
     (if (< 0 epoc)
       (if (nil? test_data)
         (println (format "Epoch %s complete" epoc))
         (println (format "Epoc %s: %s / %s" epoc (evaluate result test_data) (count test_data)))))
     (if (>= epoc epocs)
       result
       (recur (inc epoc) (time (sgd-epoc result training_data mini_batch_size eta)))))))


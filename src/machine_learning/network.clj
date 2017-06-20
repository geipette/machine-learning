(ns machine-learning.network
  (:refer-clojure :exclude [* + - /])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.random :as mr]
            [clojure.core.matrix.linear :as ml]
            [clojure.core.matrix.operators :refer [* + - /]]
            [taoensso.nippy :as nippy :refer [freeze-to-file thaw-from-file]]
            [clojure.java.io :as io])
  (:import (java.util Random)))

;(m/set-current-implementation :clatrix)
(m/set-current-implementation :vectorz)

(defn sigmoid [z]
  (if-not (or (m/vec? z) (m/matrix? z))
    (/ 1.0 (+ 1 (Math/exp (- z))))
    (m/emap sigmoid z)))

(defn sigmoid-prime [z]
  (if-not (or (m/vec? z) (m/matrix? z))
    (* (sigmoid z) (- 1 (sigmoid z)))
    (let [sz (sigmoid z)]
      (m/emul sz (m/emap #(- 1 %) sz)))))

(defprotocol Cost
  (cost [this output desired_output]
    "Return the cost associated with an output and desired output.")
  (delta [this output desired_output sigmoid-output]
    "Return the error delta from the output layer."))

(defrecord QuadraticCost []
  Cost
  (cost [this output desired_output]
    (* 0.5 (m/pow (ml/norm (- output desired_output)) 2)))
  (delta [this output desired_output sigmoid-output]
    (* (- output desired_output) (sigmoid-prime sigmoid-output))))

(defrecord CrossEntropyCost []
  Cost
  (cost [this output desired_output]
    (m/esum (- (- (m/log output)) (* (- 1 desired_output) (m/log (- 1 output))))))
  (delta [this output desired_output sigmoid-output]
    (- output desired_output)))

(defprotocol WeightInitializer
  (init-biases [this sizes]
    "Return an array of biases for the layers in the network.
   Note that the first layer is assumed to be an input layer, and
   by convention we won't set any biases for those neurons, since
   biases are only ever used in computing the outputs from later
   layers.")
  (init-weights [this sizes]
    "Return an array of weights for the neuron connections in a network."))

(defrecord LargeWeightInitializer []
  WeightInitializer
  (init-biases [this sizes]
    (map #(mr/sample-normal [% 1]) (rest sizes)))
  (init-weights [this sizes]
    (map #(mr/sample-normal [%1 %2]) (rest sizes) (butlast sizes))))

(defrecord DefaultWeightInitializer []
  WeightInitializer
  (init-biases [this sizes]
    (map #(mr/sample-normal [% 1]) (rest sizes)))
  (init-weights [this sizes]
    (map (fn [sx sy] (m/emap #(/ % (Math/sqrt sy)) (mr/sample-normal [sx sy]))) (rest sizes) (butlast sizes))))

(defn zero-array [array]
  (mapv m/zero-array (map m/shape array)))

(defrecord TrainingSpec [epocs batch_size learn_rate lambda cost])

(defn create-network
  ([sizes]
   (create-network sizes (->DefaultWeightInitializer)))
  ([sizes weight_initializer]
   (if (>= 2 (count sizes))
     (throw (IllegalArgumentException. (format "create-network requires sizes to have at least one element, was: %s" sizes))))
   {:num_layers (count sizes)
    :sizes      sizes
    :biases     (init-biases weight_initializer sizes)
    :weights    (init-weights weight_initializer sizes)}))

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
      (let [output (sigmoid (m/add (m/inner-product (first weights) result) (first biases)))]
        (recur output (rest biases) (rest weights))))))

(defn collect-activations [network input]
  (loop [activation (assert-correct-input input)
         biases (:biases network)
         weights (:weights network)
         activations [(m/matrix input)]
         zs []]
    (if (empty? biases)
      [activations zs]
      (let [z (m/add (m/inner-product (first weights) activation) (first biases))
            next_activation (sigmoid z)]
        (recur next_activation (rest biases) (rest weights) (conj activations (m/matrix next_activation)) (conj zs (m/matrix z)))))))

(defn backprop [network training_spec [input desired_output]]
  "Return a tuple '(nabla_b, nabla_w)' representing the
  gradient for the cost function C_x.  'nabla_b' and
  'nabla_w' are layer-by-layer lists of vectors."
  (let [[activations zs] (collect-activations network input)
        delta (delta (:cost training_spec) (last activations) desired_output (last zs))
        zero_b (zero-array (:biases network))
        zero_w (zero-array (:weights network))
        last_layer_index (- (:num_layers network) 2)]
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

(defn backprop-batch [network training_spec batch]
  (reduce accumulate-deltas [(zero-array (:biases network)) (zero-array (:weights network))]
          (pmap #(backprop network training_spec %) batch)))

(defn apply-delta [original delta learn_rate batch_size lambda training_data_size]
  (map (fn [x dt_x]
         (m/emap #(* (- 1 (* learn_rate (/ lambda training_data_size)))
                     (- %1 (* (/ learn_rate batch_size) %2))) x dt_x)) original delta))

(defn update-batch [network training_spec batch training_data_size]
  "Update the network's weights and biases by applying
  gradient descent using backpropagation to a single mini batch.
  The \"batch\" is a list of tuples \"(x, y)\", and \"eta\"
  is the learning rate."
  (let [[delta_biases delta_weights] (backprop-batch network training_spec batch)
        batch-size (count batch)
        {:keys [learn_rate lambda]} training_spec
        biases (apply-delta (:biases network) delta_biases learn_rate batch-size lambda training_data_size)
        weights (apply-delta (:weights network) delta_weights learn_rate batch-size lambda training_data_size)]
    (assoc network :weights weights :biases biases)))

(defn index-of-max [output]
  (first (apply max-key #(first (second %)) (map-indexed vector output))))

(defn run-test [network input]
  (index-of-max (feed-forward network input)))

(defn softmax [layer]
  "Calculates softmax distribution of a layer"
  (let [sum (m/esum layer)]
    (m/emap #(/ % sum) layer)))

(defn run-tests [network test_data]
  (pmap (fn [[input expected_output]] [(run-test network input) expected_output]) test_data))

(defn evaluate-detailed [network test_data]
  (let [test_results (apply vector (run-tests network test_data))]
    (reduce-kv (fn [result pos [x y]] (if (= x y) result (conj result pos))) [] test_results)))

(defn evaluate [network test_data]
  (- (count test_data) (count (evaluate-detailed network test_data))))

(defn sgd-epoc [network training_spec training_data]
  (let [batch_size (:batch_size training_spec)
        training_data_size (count training_data)
        batches (partition batch_size batch_size nil (shuffle training_data))]
    (loop [updated_network network remaining_batches batches]
      (if (empty? remaining_batches)
        updated_network
        (recur (update-batch updated_network training_spec (first remaining_batches) training_data_size) (rest remaining_batches))))))

(defn sgd [network training_spec training_data test_data]
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
    (if (>= epoc (:epocs training_spec))
      result
      (recur (inc epoc)
             (sgd-epoc result training_spec training_data)))))

(defn save-network [network filename]
  (freeze-to-file (io/file filename) network))

(defn load-network [filename]
  (thaw-from-file (io/file filename)))
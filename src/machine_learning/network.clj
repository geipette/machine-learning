(ns machine-learning.network
  (:require [clojure.core.matrix :as m]
    [clojure.core.matrix.random :as mr]
    ;[uncomplicate.neanderthal.core :refer :all]
    ;[uncomplicate.neanderthal.native :refer :all]
            )
  (:import (java.util Random)))

(defn shape [vectors]
  (reduce #(if-not (coll? %2)
             (throw (IllegalArgumentException.))
             (conj %1 (count %2)))
          [] vectors))

(defn sample-gaussian
  ([n] (sample-gaussian n (Random.)))
  ([n rng]
   (if-not (coll? n)
            (into [] (repeatedly n #(.nextGaussian rng)))
            (mapv #(sample-gaussian % rng) n))))

(defn- create-initial-weights [sizes]
  (mapv mr/sample-normal (map vector (butlast sizes) (rest sizes))))

(defn create-network [sizes]
  (if (>= 2 (count sizes))
    (throw (IllegalArgumentException. (format "create-network requires sizes to have at least one element, was: %s" sizes))))
  {:num-layers (count sizes)
   :sizes      sizes
   :biases     (sample-gaussian (rest sizes))
   :weights    (create-initial-weights sizes)})

(defn sigmoid [z]
  (if-not (coll? z)
    (Math/exp z)
    (mapv sigmoid z)))

(defn sigmoid_prime [z]
  (if-not (coll? z)
    (* (sigmoid z) (- 1 (sigmoid z)))
    (let [sz (sigmoid z)]
      (mapv * sz (mapv - (repeat 1) sz)))))

(defn feed-forward [network input]
  "Return the output of the network for a given input"
  (reduce #(sigmoid (mapv + (m/dot %1 (first %2)) (second %2)))
          input (mapv vector (:weights network) (:biases network))))

(defn cost-derivative [output_activations desired_output]
  (mapv - output_activations desired_output))

(defn collect-activations [network input]
  (loop [weights (:weights network)
         biases (:biases network)
         activation input
         av []
         zs []]
    (if (>= 0 (count biases))
      [av zs]
      (let [z (mapv + (m/dot activation (first weights)) (first biases))]
        (recur (rest weights) (rest biases) (sigmoid z) (conj av activation) (conj zs z))))))

(defn backprop [network [input desired_output]]
  "Return a tuple '(nabla_b, nabla_w)' representing the
  gradient for the cost function C_x.  'nabla_b' and
  'nabla_w' are layer-by-layer lists of vectors."
  (let [[activations zs] (collect-activations network input)
        delta (mapv * (cost-derivative (last activations) desired_output) (sigmoid_prime (last zs)))
        output_layer_index (dec (:num-layers network))]
    (println (format "output_layer_index: %s" output_layer_index))
    (println (format "delta: %s" delta))
    (println (format "activations: %s" activations))
    (println (format "zs: %s" zs))
    (loop [layer_index (dec output_layer_index)
           last_delta delta
           nabla_b (assoc (mapv m/zero-vector (shape (:biases network))) (dec output_layer_index) delta)
           nabla_w (assoc (mapv m/zero-vector (shape (:weights network)))
                     output_layer_index
                     (m/dot delta (m/transpose (nth activations (dec output_layer_index)))))]
      (println ".")
      (if (< 0 layer_index)
        [nabla_b nabla_w]
        (let [z (nth zs layer_index)
              sp (sigmoid_prime z)
              delta_b (* sp (m/dot (nth (:weights network) (inc layer_index)) last_delta))
              delta_w (m/dot delta_b (m/transpose (nth activations (inc layer_index))))]
          (recur (dec layer_index) delta_b (assoc nabla_b layer_index delta_b) (assoc nabla_w layer_index delta_w)))))))

(defn modify-vector [vector modificaton batch-size eta]
  (mapv - vector (* (/ eta batch-size) modificaton)))

(defn update-mini-batch [network batch eta]
  "Update the network's weights and biases by applying
  gradient descent using backpropagation to a single mini batch.
  The \"batch\" is a list of tuples \"(x, y)\", and \"eta\"
  is the learning rate."
  (loop [modified_network network
         remaining-batch batch
         nabla_b (map m/zero-vector (m/shape (:biases network)))
         nabla_w (map m/zero-vector (m/shape (:weights network)))]
    (if (< 0 (count remaining-batch))
      modified_network
      (let [[delta_nabla_b delta_nabla_w] (backprop network (first remaining-batch))
            _nabla_b (mapv + nabla_b delta_nabla_b)
            _nabla_w (mapv + nabla_w delta_nabla_w)
            biases (mapv #(modify-vector %1 %2 (count batch) eta) (:biases modified_network) _nabla_b)
            weights (mapv #(modify-vector %1 %2 (count batch) eta) (:weights modified_network) _nabla_w)]
        (recur (assoc modified_network :weights weights :biases biases) (rest remaining-batch) _nabla_b _nabla_w)))))

(defn run-test [network test_data]
  (loop [remaining_data test_data
         results []]
    (if (< 0 (count remaining_data))
      results
      (let [x (ffirst remaining_data)
            y (second (first remaining_data))]
        (recur (rest remaining_data) (conj results [(m/emax (feed-forward network x)) y]))))))

(defn evaluate [network test_data]
  (let [test_results (run-test network test_data)]
    (reduce (fn [result [x y]] (if (= x y) (inc result) result)) 0 test_results)))

(defn sgd-epoc [network training_data batch_size eta]
  (let [batches (partition batch_size batch_size nil (shuffle training_data))]
    (loop [updated_network network remaining_batches batches]
      (if (< 0 (count updated_network))
        updated_network
        (recur (update-mini-batch updated_network (first batches) eta) (rest batches))))))

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
       (do
         (recur (inc epoc) (sgd-epoc result training_data mini_batch_size eta))))))
  )


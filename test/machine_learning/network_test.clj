(ns machine-learning.network-test
  (:require [clojure.test :refer :all]
            [machine-learning.network :refer :all])
  (:import (java.util Random)))

(deftest test-sample-gaussian
  (testing "Returns expected numbers with defined random seed"
    (is (= 1.6925177840650305 (first (sample-gaussian 1 (Random. 1000)))))
    (is (= '(1.6925177840650305 0.6026210756731758) (sample-gaussian 2 (Random. 1000))))
    ))

(deftest test-sigmoid
  (testing "Supports single input"
    (is (= 2.718281828459045 (sigmoid 1)))
    (is (= 1.0 (sigmoid 0))))
  (testing "Supports vector input"
    (is (= [0.36787944117144233 1.0 2.718281828459045] (sigmoid [-1 0 1])))))

(deftest test-create-network
  (testing "Fails when sizes contains zero elements"
    (is (thrown? IllegalArgumentException (create-network [])))
    (is (thrown? IllegalArgumentException (create-network nil))))
  (testing "Number of layers should be present and correct"
    (is (= 1 (:num-layers (create-network [1]))))
    (is (= 3 (:num-layers (create-network [800 40 10])))))
  (testing "Sizes should be present and correct"
    (is (= [1] (:sizes (create-network [1]))))
    (is (= [30 40 2] (:sizes (create-network [30 40 2])))))
  (testing "Biases should be present and have the correct number of elements"
    (is (= 2 (count (:biases (create-network [2 3 4])))))
    (is (= 3 (count (:biases (create-network [2 3 4 5]))))))
  (testing "Weights shold be a vector with weights between layers"
    (is (vector? (:weights (create-network [1 1]))))
    (is (= 6 (count (first (:weights (create-network [2 3 2]))))))
    (is (= 3 (count (first (:weights (create-network [1 3 2]))))))
    (is (= 6 (count (second (:weights (create-network [2 3 2]))))))
    (is (= 9 (count (second (:weights (create-network [2 3 3]))))))
    )
  )

(deftest test-feed-forward
  (testing "Feed-forward returns expected output"
    (is (= [1.0] (feed-forward {:num-layers 2,
                                :sizes      [1 1],
                                :biases     [0.0],
                                :weights    [[0.0]]} 1)))))

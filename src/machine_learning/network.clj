(ns machine-learning.network
  )

(defn create-network [sizes]
  {:num-layers (alength sizes)
   :sizes sizes})

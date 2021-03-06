(defproject machine-learning "0.1.0-SNAPSHOT"
  :description "Testing machine learning in clojure"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/tools.logging "0.3.1"]
                 [ch.qos.logback/logback-classic "1.2.3"]
                 [net.mikera/core.matrix "0.60.3"]
                 [net.mikera/vectorz-clj "0.47.0"]
                 [clatrix "0.5.0"]
                 [uncomplicate/neanderthal "0.10.0"]
                 [org.clojure/data.json "0.2.6"]
                 [org.clojure/tools.cli "0.3.5"]
                 [metrics-clojure "2.9.0"]
                 [com.taoensso/nippy "2.13.0"]
                 [hiccup "1.0.5"]]

  :main ^:skip-aot machine-learning.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}
             :dev {:dependencies [[midje "1.8.3"]]}}
  :plugins [[lein-midje "3.2.1"]]
  :jvm-opts ["-Xmx8g"]
  :aliases {"process-resources" ["run" "-m" "machine-learning.mnist-reader/process-resources"]})

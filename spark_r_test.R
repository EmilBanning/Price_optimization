########### install spark

library(sparklyr)
library(devtools)
spark_install(version = "2.1.0")
#devtools::install_github("rstudio/sparklyr")

########### create spark instance

library(sparklyr)
sc <- spark_connect(master = "local")

########### using dplyr

install.packages(c("nycflights13", "Lahman"))

library(dplyr)
iris_tbl <- copy_to(sc, iris)
flights_tbl <- copy_to(sc, nycflights13::flights, "flights")
batting_tbl <- copy_to(sc, Lahman::Batting, "batting")
src_tbls(sc)

flights_tbl %>% filter(dep_delay == 2)


batting_tbl %>%
  select(playerID, yearID, teamID, G, AB:H) %>%
  arrange(playerID, yearID, teamID) %>%
  group_by(playerID) %>%
  filter(min_rank(desc(H)) <= 2 & H > 0)


########## using SQL

library(DBI)
iris_preview <- dbGetQuery(sc, "SELECT * FROM iris LIMIT 10")
iris_preview


########### using Spark ML

# copy mtcars into spark
mtcars_tbl <- copy_to(sc, mtcars)

# transform our data set, and then partition into 'training', 'test'
partitions <- mtcars_tbl %>%
  filter(hp >= 100) %>%
  mutate(cyl8 = cyl == 8) %>%
  sdf_partition(training = 0.5, test = 0.5, seed = 1099)

# fit a linear model to the training dataset
fit <- partitions$training %>%
  ml_linear_regression(response = "mpg", features = c("wt", "cyl"))
fit
summary(fit)

# above is similar to std linear regression
head(mtcars_tbl)
head(partitions)
linear_regression <- lm(mpg ~ wt + cyl, data = partitions$training)
linear_regression
summary(linear_regression)

########### reading and writing data

temp_csv <- tempfile(fileext = ".csv")
temp_parquet <- tempfile(fileext = ".parquet")
temp_json <- tempfile(fileext = ".json")

spark_write_csv(iris_tbl, temp_csv)
iris_csv_tbl <- spark_read_csv(sc, "iris_csv", temp_csv)

spark_write_parquet(iris_tbl, temp_parquet)
iris_parquet_tbl <- spark_read_parquet(sc, "iris_parquet", temp_parquet)

spark_write_json(iris_tbl, temp_json)
iris_json_tbl <- spark_read_json(sc, "iris_json", temp_json)

src_tbls(sc)


########## using Spark for distributed R

spark_apply(iris_tbl, function(data) {
  data[1:4] + rgamma(1,2)
})


spark_apply(
  iris_tbl,
  function(e) broom::tidy(glm(Petal_Width ~ Petal_Length, data = e, family = "gaussian")),
  names = c("term", "estimate", "std.error", "statistic", "p.value"),
  group_by = "Species"
)


########## 


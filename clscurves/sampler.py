class SimpleSampler:
    """
    A simple PySpark DataFrame down-sampler. The typical ".sample"
    DF method only lets you sample a certain approximate fraction
    of rows from a DataFrame, with no guarantees on the number of
    rows that will get sampled. This class will sample an exact
    number of rows without replacement from an input DataFrame,
    returning the original DF if it has fewer rows than the
    desired sample size.
    """

    def __init__(self):
        pass

    def get_sample(
            self,
            df: DataFrame,
            max_size: int = 100000,
            seed: int = 1) -> DataFrame:
        """
        Get a random DataFrame sample of definite maximum size.
        PARAMETERS
        ----------
          df -- input DataFrame.
          max_size -- max number of rows to sample (default: 100000).
          seed -- random seed for sample (default: 1).
        RETURNS
        -------
          df -- sampled DataFrame.
        """
        num_examples = df.count()
        if num_examples > max_size:
            print(
                "%d exceeds maximum number of examples (%d). Will sample to %d." % (
                    num_examples, max_size, max_size))
            df = df \
                .orderBy(F.rand(seed)) \
                .limit(max_size)

        return df
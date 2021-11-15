import tensorflow as tf


class Autoencoder_Classification_Net(tf.keras.Model):
    """Transformer MLP / feed-forward block."""

    # mlp_dim: int
    # dtype: Dtype = jnp.float32
    # out_dim: Optional[int] = None
    # dropout_rate: float = 0.1
    # kernel_init: Callable[[PRNGKey, Shape, Dtype],
    #                         Array] = nn.initializers.xavier_uniform()
    # bias_init: Callable[[PRNGKey, Shape, Dtype],
    #                     Array] = nn.initializers.normal(stddev=1e-6)
    
    def __init__(self, mlp_dim = 10, layer_num = 10, out_dim = 10, name = None):
        super(Autoencoder_Classification_Net, self).__init__(name)
        # actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.layer_num = layer_num

        self.d1 = tf.keras.layers.Dense(self.mlp_dim)
        self.layer_list = []
        # for i in range(layer_num):
        #     self.layer_list.append(tf.keras.layers.Dense(self.mlp_dim, activation = "tanh"))
        self.c1 = tf.keras.layers.Conv2D(mlp_dim*(2), 3, 1, padding = "same", activation="tanh") # =>28, 28
        self.c2 = tf.keras.layers.Conv2D(mlp_dim*(2), 4, 2, padding = "same", activation="tanh") # =>14, 14
        self.c3 = tf.keras.layers.Conv2D(mlp_dim*(2**2), 3, 1, padding = "same", activation="tanh") # =>14, 14
        self.c4 = tf.keras.layers.Conv2D(mlp_dim*(2**2), 4, 2, padding = "same", activation="tanh") # =>7, 7
        self.c5 = tf.keras.layers.Conv2D(mlp_dim*(2**3), 3, 1, padding = "same", activation="tanh") # =>7, 7
        self.c6 = tf.keras.layers.Conv2D(mlp_dim*(2**4), 7, 1, padding = "valid", activation="tanh") # => 1, 1 
        self.f1 = tf.keras.layers.Flatten()

        self.head = tf.keras.layers.Dense(self.out_dim, name = "cls")

        self.uf1 = tf.keras.layers.Reshape([1,1,mlp_dim*(2**4)])
        self.c7 = tf.keras.layers.Conv2DTranspose(mlp_dim*(2**4), 7, 1, padding = "valid", activation="tanh") # => 1, 1 
        self.c8 = tf.keras.layers.Conv2D(mlp_dim*(2**3), 3, 1, padding = "same", activation="tanh") # =>7, 7
        self.c9 = tf.keras.layers.Conv2DTranspose(mlp_dim*(2**2), 4, 2, padding = "same", activation="tanh") # =>7, 7
        self.c10 = tf.keras.layers.Conv2D(mlp_dim*(2**2), 3, 1, padding = "same", activation="tanh") # =>14, 14
        self.c11 = tf.keras.layers.Conv2DTranspose(mlp_dim*(2), 4, 2, padding = "same", activation="tanh") # =>14, 14
        self.c12 = tf.keras.layers.Conv2D(1, 3, 1, padding = "same", activation="tanh", name = "recon") # =>28, 28
        


        
    def call(self, inputs):
        '''
        x: (32, 28, 28, 128)
        x: (32, 14, 14, 128)
        x: (32, 14, 14, 256)
        x: (32, 7, 7, 256)
        x: (32, 7, 7, 512)
        x: (32, 1, 1, 1024)
        embed: (32, 1024)
        x: (32, 1, 1, 1024)
        x: (32, 7, 7, 1024)
        x: (32, 7, 7, 512)
        x: (32, 14, 14, 256)
        x: (32, 14, 14, 256)
        x: (32, 28, 28, 128)

        recon_output = (32, 28, 28, 1)

        
        '''
        x = inputs
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        embed = self.f1(x)
        print("embed:",embed.shape)
        x = self.uf1(embed)
        x = self.c7(x)
        x = self.c8(x)
        x = self.c9(x)
        x = self.c10(x)
        x = self.c11(x)

        recon_output = (self.c12(x)+1)/2 # to 0~1

        cls_output = self.head(embed)


        return [cls_output, recon_output]

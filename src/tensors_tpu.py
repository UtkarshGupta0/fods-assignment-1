import time

import numpy as np
import tensorflow as tf


def part1_tensor_ranks():
    print("\n=== Part 1: tensors by rank ===")
    examples = {
        "scalar  (rank 0)": tf.constant(3.14),  # a single number
        "vector  (rank 1)": tf.constant([1.0, 2.0, 3.0]),  # one MFCC frame
        "matrix  (rank 2)": tf.constant([[1.0, 2.0], [3.0, 4.0]]),  # grayscale image, TF-IDF matrix
        "3-D     (rank 3)": tf.zeros([224, 224, 3]),  # one RGB image  (H, W, C)
        "4-D     (rank 4)": tf.zeros([8, 224, 224, 3]),  # batch of images (N, H, W, C)
        "5-D     (rank 5)": tf.zeros([2, 16, 224, 224, 3]),  # batch of videos (N, T, H, W, C)
    }
    for name, t in examples.items():
        print(f"  {name}: shape={tuple(t.shape)}  rank={tf.rank(t).numpy()}  dtype={t.dtype.name}  "
              f"elements={t.shape.num_elements()}")

    print("\n  Natural tensor shape per modality used in this assignment:")
    modal = [
        ("RGB image batch", (32, 224, 224, 3), "float32", "(batch, height, width, channels)"),
        ("Multispectral batch (EuroSAT)", (32, 64, 64, 13), "float32", "13 Sentinel-2 bands as channels"),
        ("Hyperspectral cube (Indian Pines)", (145, 145, 200), "float32", "(H, W, bands): bands = spectrum axis"),
        ("Thermal image batch", (32, 512, 640, 1), "float32", "1 channel = temperature/intensity"),
        ("RGB video batch", (8, 16, 112, 112, 3), "float32", "(batch, time, H, W, C)"),
        ("Log-mel spectrogram batch", (32, 64, 300, 1), "float32", "(batch, mel bins, frames, 1)"),
        ("wav2vec2 hidden states", (1, 180, 768), "float32", "(batch, frames, d_model)"),
        ("Token embeddings (BERT)", (32, 128, 768), "float32", "(batch, seq_len, d_model)"),
    ]
    print(f"  {'modality':<36}{'shape':<26}{'MB (float32)':>13}   meaning")
    for name, shape, dt, meaning in modal:
        mb = np.prod(shape) * 4 / 1e6
        print(f"  {name:<36}{str(shape):<26}{mb:>13.1f}   {meaning}")


def part2_operations():
    print("\n=== Part 2: operations and autodiff ===")
    a = tf.constant([[1., 2., 3.], [4., 5., 6.]])  # (2, 3)
    b = tf.constant([[1., 0.], [0., 1.], [1., 1.]])  # (3, 2)
    print("  matmul (2x3)@(3x2) ->", tf.matmul(a, b).numpy().tolist())
    print("  broadcasting: (2,3) + (3,) ->", (a + tf.constant([10., 20., 30.])).numpy().tolist())
    print("  reshape (2,3)->(3,2):", tf.reshape(a, (3, 2)).numpy().tolist())
    print("  reduce_mean over axis 0:", tf.reduce_mean(a, axis=0).numpy())
    # einsum names the axes explicitly: batched image-patch similarity  (n,h,d) x (n,h,d) -> (n,h)
    x, y = tf.random.normal([4, 5, 8]), tf.random.normal([4, 5, 8])
    print("  einsum 'nhd,nhd->nh' shape:", tf.einsum("nhd,nhd->nh", x, y).shape)
    # Automatic differentiation: GradientTape records operations and applies the chain rule
    w = tf.Variable([[0.5], [-0.3]])
    xin, target = tf.constant([[1.0, 2.0]]), tf.constant([[1.0]])
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((tf.matmul(xin, w) - target) ** 2)
    print(f"  d(loss)/dw via GradientTape = {tape.gradient(loss, w).numpy().ravel()}  (loss={loss.numpy():.3f})")


def part3_sparse_ragged():
    print("\n=== Part 3: sparse and ragged tensors ===")
    # TF-IDF documents are >99% zeros: store only the non-zero entries (indices + values)
    sp = tf.sparse.SparseTensor(indices=[[0, 2], [1, 0], [1, 3]], values=[0.7, 0.4, 0.9], dense_shape=[2, 5])
    print("  sparse TF-IDF-like matrix dense form:", tf.sparse.to_dense(sp).numpy().tolist())
    rt = tf.ragged.constant([[1, 2, 3], [4], [5, 6]])
    print("  ragged shape:", rt.shape, "-> row lengths:", rt.row_lengths().numpy(),
          "-> padded dense shape:", rt.to_tensor().shape)


def part4_precision():
    print("\n=== Part 4: float32 vs bfloat16 ===")
    v = tf.constant([1.0001, 3.14159265, 1e-8, 3e38])
    print("  float32 :", v.numpy())
    print("  bfloat16:", tf.cast(tf.cast(v, tf.bfloat16), tf.float32).numpy(), " (note lost digits, same range)")


def part5_xla():
    print("\n=== Part 5: XLA (Accelerated Linear Algebra) ===")
    def f(x, w, b):
        return tf.nn.gelu(tf.matmul(x, w) + b) * 2.0 + 1.0

    f_graph = tf.function(f)
    f_xla = tf.function(f, jit_compile=True)
    x, w, b = tf.random.normal([512, 1024]), tf.random.normal([1024, 1024]), tf.random.normal([1024])

    def bench(fn, n=30):
        fn(x, w, b).numpy()
        t = time.perf_counter()
        for _ in range(n):
            out = fn(x, w, b)
        out.numpy()
        return (time.perf_counter() - t) / n * 1e3

    print(f"  graph mode: {bench(f_graph):7.2f} ms/call   XLA jit_compile: {bench(f_xla):7.2f} ms/call "
          f"({'CPU' if not tf.config.list_physical_devices('GPU') else 'GPU'}; gains are largest on TPU/GPU)")
    print("  results match:", np.allclose(f_graph(x, w, b), f_xla(x, w, b), atol=1e-3))
    try:
        hlo = f_xla.experimental_get_compiler_ir(x, w, b)(stage="hlo")
        lines = [l.strip() for l in hlo.splitlines() if l.strip()]
        print(f"  HLO program has {len(lines)} lines; first lines:")
        for l in lines[:6]:
            print("     ", l[:110])
    except Exception as e:
        print("  (HLO dump not available here:", type(e).__name__, ")")


def get_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="")
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.TPUStrategy(resolver), "TPU"
    except Exception as e:
        print(f"  no TPU available ({type(e).__name__}); falling back")
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy(), f"{len(gpus)} GPUs"
    return tf.distribute.get_strategy(), ("1 GPU" if gpus else "CPU")


def part6_strategy():
    print("\n=== Part 6: distribution strategy ===")
    strategy, kind = get_strategy()
    print(f"  device: {kind}   replicas in sync: {strategy.num_replicas_in_sync}")
    global_batch = 64 * strategy.num_replicas_in_sync
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Input((32, 32, 3)),
            tf.keras.layers.Conv2D(16, 3, activation="relu"), tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10)])
        model.compile("adam", tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])
    X = np.random.rand(global_batch * 4, 32, 32, 3).astype("float32")
    y = np.random.randint(0, 10, global_batch * 4)
    ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(global_batch, drop_remainder=True)
    hist = model.fit(ds, epochs=2, verbose=0)
    print(f"  trained 2 epochs on {kind}; final loss={hist.history['loss'][-1]:.3f}")


if __name__ == "__main__":
    print("TensorFlow", tf.__version__)
    part1_tensor_ranks();
    part2_operations();
    part3_sparse_ragged()
    part4_precision();
    part5_xla();
    part6_strategy()

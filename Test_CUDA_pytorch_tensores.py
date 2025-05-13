# Importamos las librerías torch, numpy y time
import torch
import numpy as np
import time

# Comprobamos si hay GPU
def check_gpu():
    if torch.cuda.is_available():
        return True
    else:
        return False
    
# Comprobamos si hay GPU de otra forma
if __name__ == "__main__":
    if check_gpu():
        print("Hay GPU disponible.")
    else:
        print("No hay GPU disponible.")

# Comprobamos si hay GPU y Torch
def check_torch():
    try:
        import torch
        return True
    except ImportError:
        return False
    
# Comprobamos si hay GPU y Torch de otra forma
if __name__ == "__main__":
    if check_torch():
        print("Torch está instalado.")
    else:
        print("Torch no está instalado.")

# Vamos a comprobar si veo la GPU de otra forma
def check_gpu_visible():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            return True
        else:
            return False
    else:
        return False
    
# Comprobamos si hay GPU visible de otra forma
if __name__ == "__main__":
    if check_gpu_visible():
        print("La GPU es visible.")
    else:
        print("La GPU no es visible.")

# Comprobamos si hay GPU visible y Torch de otra forma
def check_torch_visible():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                return True
            else:
                return False
        else:
            return False
    except ImportError:
        return False
    
# Comprobamos si hay GPU visible y Torch de otra forma
if __name__ == "__main__":
    if check_torch_visible():
        print("Torch está instalado y la GPU es visible.")
    else:
        print("Torch no está instalado o la GPU no es visible.")
      
# Mas info sobre la GPU
def check_torch_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                for i in range(gpu_count):
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                return True
            else:
                return False
        else:
            return False
    except ImportError:
        return False
    
# Comprobamos si hay GPU visible y puedo obtener información de ella y Torch
if __name__ == "__main__":
    if check_torch_gpu_info():
        print("Torch está instalado y la GPU es visible y puedo obtener información de ella.")
    else:
        print("Torch no está instalado o la GPU no es visible o no puedo obtener información de ella.")
print()

# Comparamos calculo con CPU y GPU para la suma de los elementos de un array
# Generamos un array aleatorio
array_size = 1_000_000_000
array = np.random.rand(array_size)

# Definimos una función para calcular la suma de los elementos del array
def sum_array(array):
    return np.sum(array)

# Medimos el tiempo de ejecución en CPU
start_time = time.time()
sum_cpu = sum_array(array)
end_time = time.time()
time_cpu = end_time - start_time
print(f"Suma CPU: {sum_cpu}, Tiempo: {time_cpu:.4f} segundos")

# Medimos el tiempo de ejecución en GPU
array_gpu = torch.tensor(array).cuda()
start_time = time.time()
sum_gpu = torch.sum(array_gpu)
end_time = time.time()
time_gpu = end_time - start_time
print(f"Suma GPU: {sum_gpu.item()}, Tiempo: {time_gpu:.4f} segundos")

# Ratio de velocidad
if time_gpu > 0:
    ratio = (time_cpu / time_gpu)
    print(f"Ratio de velocidad GPU/CPU: {ratio:.4f}")
else:
    print("No se puede calcular el ratio de velocidad, el tiempo de GPU es 0.")
print()
    
# Comprobamos calculo con CPU y GPU con tensores

# Generamos un tensor aleatorio de tamaño muy grande
array_size = 200
array_1 = np.random.rand(array_size, array_size, array_size)
# Imprimimos el tamaño de el shape del array
print(array_1.shape)
print()
# Convertimos el array a tensor con torch
tsr_1 = torch.tensor(array_1, dtype=torch.float32)
# Imprimimos el rango del tensor    
print(tsr_1.shape)
print()

# Generamos otro tensor aleatorio de tamaño muy grande
array_size = 200
array_2 = np.random.rand(array_size, array_size, array_size)
# Imprimimos el tamaño de el shape del array
print(array_2.shape)
print()
# Convertimos el array a tensor con torch
tsr_2 = torch.tensor(array_2, dtype=torch.float32)
# Imprimimos el rango del tensor    
print(tsr_2.shape)
print()

# Comprobamos que ambos son tensores
is_tsr_1_a_tensor = (isinstance(tsr_1, torch.Tensor))
print("tsr_1 is a tensor: ", is_tsr_1_a_tensor)
is_tsr_2_a_tensor = (isinstance(tsr_2, torch.Tensor))
print("tsr_2 is a tensor: ", is_tsr_2_a_tensor)
print()

# Definimos una función para calcular el producto punto de los tensores
def dot_product(tensor1, tensor2):
    result = torch.dot(tensor1.flatten(), tensor2.flatten())
    return (result)

# Medimos el tiempo de ejecución en CPU
start_time = time.time()
dot_product_cpu = dot_product(tsr_1, tsr_2)
end_time = time.time()
time_cpu = end_time - start_time

print("Producto dot CPU: ", dot_product_cpu.item())
print("Tiempo de CPU: ", time_cpu, "segundos")

# Medimos el tiempo de ejecución en GPU
tsr_1_gpu = tsr_1.cuda()
tsr_2_gpu = tsr_2.cuda()
start_time = time.time()
dot_product_gpu = dot_product(tsr_1_gpu, tsr_2_gpu)
end_time = time.time()
time_gpu = end_time - start_time
print("Producto dot GPU: ", dot_product_gpu.item())
print("Tiempo de GPU: ", time_gpu, "segundos")

# Ratio de velocidad
if time_gpu > 0:
    ratio = (time_cpu / time_gpu)
    print(f"Ratio de velocidad GPU/CPU: {ratio:.4f}")
else:
    print("No se puede calcular el ratio de velocidad, el tiempo de GPU es 0.")
print()

# Definimos una función para calcular el producto bbm de los tensores
def tsr_product_tensors(tensor1, tensor2):
    result = torch.bmm(tensor1, tensor2)
    return (result)

# Medimos el tiempo de ejecución en CPU
start_time = time.time()
tsr_product_cpu = dot_product(tsr_1, tsr_2)
end_time = time.time()
time_cpu = end_time - start_time
print("Producto bmm CPU: ", tsr_product_cpu.item())
print("Tiempo de CPU: ", time_cpu, "segundos")

# Medimos el tiempo de ejecución en GPU
tsr_1_gpu = tsr_1.cuda()
tsr_2_gpu = tsr_2.cuda()
start_time = time.time()
tsr_product_gpu = dot_product(tsr_1_gpu, tsr_2_gpu)
end_time = time.time()
time_gpu = end_time - start_time
print("Producto bmm GPU: ", tsr_product_gpu.item())
print("Tiempo de GPU: ", time_gpu, "segundos")

# Ratio de velocidad
if time_gpu > 0:
    ratio = (time_cpu / time_gpu)
    print(f"Ratio de velocidad GPU/CPU: {ratio:.4f}")
else:
    print("No se puede calcular el ratio de velocidad, el tiempo de GPU es 0.")
print()

# Definimos una función para calcular el producto @ de los tensores
def tsr_product_tensors(tensor1, tensor2):
    result = tensor1@tensor2
    return (result)

# Medimos el tiempo de ejecución en CPU
start_time = time.time()
tsr_product_cpu = dot_product(tsr_1, tsr_2)
end_time = time.time()
time_cpu = end_time - start_time
print("Producto @ CPU: ", tsr_product_cpu.item())
print("Tiempo de CPU: ", time_cpu, "segundos")

# Medimos el tiempo de ejecución en GPU
tsr_1_gpu = tsr_1.cuda()
tsr_2_gpu = tsr_2.cuda()
start_time = time.time()
tsr_product_gpu = dot_product(tsr_1_gpu, tsr_2_gpu)
end_time = time.time()
time_gpu = end_time - start_time
print("Producto @ GPU: ", tsr_product_gpu.item())
print("Tiempo de GPU: ", time_gpu, "segundos")

# Ratio de velocidad
if time_gpu > 0:
    ratio = (time_cpu / time_gpu)
    print(f"Ratio de velocidad GPU/CPU: {ratio:.4f}")
else:
    print("No se puede calcular el ratio de velocidad, el tiempo de GPU es 0.")
print()

# Definimos una función para calcular el producto matmult de los dos tensores
def tsr_product(tensor1, tensor2):
    result = torch.matmul(tensor1, tensor2)
    return (result)

# Medimos el tiempo de ejecución en CPU
start_time = time.time()
tsr_product_cpu = dot_product(tsr_1, tsr_2)
end_time = time.time()
time_cpu = end_time - start_time
print("Producto matmul CPU: ", tsr_product_cpu.item())
print("Tiempo de CPU: ", time_cpu, "segundos")

# Medimos el tiempo de ejecución en GPU
tsr_1_gpu = tsr_1.cuda()
tsr_2_gpu = tsr_2.cuda()
start_time = time.time()
tsr_product_gpu = dot_product(tsr_1_gpu, tsr_2_gpu)
end_time = time.time()
time_gpu = end_time - start_time
print("Producto matmul GPU: ", tsr_product_gpu.item())
print("Tiempo de GPU: ", time_gpu, "segundos")

# Ratio de velocidad
if time_gpu > 0:
    ratio = (time_cpu / time_gpu)
    print(f"Ratio de velocidad GPU/CPU: {ratio:.4f}")
else:
    print("No se puede calcular el ratio de velocidad, el tiempo de GPU es 0.")
print()
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    # Para obtener la memoria total de tu GPU de Windows:
    # Crea un tensor pequeño para que PyTorch inicialice el contexto CUDA
    dummy_tensor = torch.randn(1).cuda()
    print(torch.cuda.get_device_properties(0).total_memory / (1024**3)) # Memoria total en GB
exit() # Para salir del intérprete de Python
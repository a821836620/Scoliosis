import timm

model_names = timm.list_models('*swin*')
print(model_names)
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
num_features = model.head.in_features
model.head = nn.Linear(num_features, 5)
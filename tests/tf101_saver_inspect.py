# inspect_checkpoint 라이브러리 불어오기
from tensorflow.python.tools import inspect_checkpoint as chkp

# checkpoint 파일 안의 모든 텐서를 출력
chkp.print_tensors_in_checkpoint_file("./tmp/model.ckpt", tensor_name='', all_tensors=True)

# checkpoint 파일 안의 "v1" 키의 텐서만 출력
chkp.print_tensors_in_checkpoint_file("./tmp/model.ckpt", tensor_name='v1', all_tensors=False)

# checkpoint 파일 안의 "v2" 키의 텐서만 출력
chkp.print_tensors_in_checkpoint_file("./tmp/model.ckpt", tensor_name='v2', all_tensors=False)
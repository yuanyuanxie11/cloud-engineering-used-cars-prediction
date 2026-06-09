FROM public.ecr.aws/lambda/python:3.12

COPY serving/requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# modeling.features + modeling.train are reused so serve-time preprocessing
# matches train-time exactly. config.yaml supplies the feature schema.
COPY modeling/ ${LAMBDA_TASK_ROOT}/modeling/
COPY config.yaml ${LAMBDA_TASK_ROOT}/config.yaml
COPY serving/handler.py ${LAMBDA_TASK_ROOT}/handler.py

CMD ["handler.handler"]

FROM ldm_base

WORKDIR /app
RUN echo $PATH
SHELL ["conda", "run", "-n", "ldm", "/bin/bash", "-c"]
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development
ENV TRANSFORMERS_CACHE="/app/models/cache"
COPY templates /app/templates/
COPY app.py /app/
RUN conda install -n ldm flask
EXPOSE 5000
CMD ["conda", "run", "-n", "ldm", "flask", "run"]
# german-dpo
A set of german pdo dataset translations and tools to create them

# Prepare configuration
```
cp .env.dev .env
```
Add, if necessary, the API_KEY and change the models according to your requirements

# Create new environment
```
make build
```

# Start app
```
make up
```

# Estimates the costs/runtime of the system & user prompt translation
```
make estimate_translation
```

# Estimates the costs/runtime of the inference task for the 'chosen' (will be later extended with the 'rejected') field
```
make estimate_inference
```

# Starts the system & user prompt translation 
```
make run_translations
```

# Starts the inference for the 'chosen' (will be later extended with the 'rejected') field 
```
make run_inference
```

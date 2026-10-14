figures_3t = ["figure_04", "figure_05", "figure_06", "figure_07", "figure_08"]

data_3t = expand(
    "data/3T_temperature_recordings/{name}.txt",
    name=glob_wildcards("data/3T_temperature_recordings/{name}.txt").name,
)

rule all:
    input:
        expand("figures/figure_03.{ext}", ext=["svg", "png"]),
        expand("figures/{fig}.{ext}", fig=figures_3t, ext=["svg", "png"]),
        expand("results/{fig}_metrics.csv", fig=figures_3t),


rule analysis_7T:
    input:
        script="code/01_7T_analysis.py",
        data="data/7T_temperature_recordings/7T_heating_slew_rates.csv",
    output:
        expand("figures/figure_03.{ext}", ext=["svg", "png"]),
    shell:
        "uv run {input.script}"


rule analysis_3T:
    input:
        script="code/02_3T_analysis.py",
        data=data_3t,
    output:
        expand("figures/{fig}.{ext}", fig=figures_3t, ext=["svg", "png"]),
        expand("results/{fig}_metrics.csv", fig=figures_3t),
    shell:
        "uv run {input.script}"

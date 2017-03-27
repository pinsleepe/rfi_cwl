cwlVersion: v1.0
class: CommandLineTool
label: Example trivial wrapper for python compiler
hints:
  DockerRequirement:
    dockerPull: pulsar_docker
baseCommand: python
inputs:
  src:
    type: File
    inputBinding:
      position: 1
outputs:
  classfile:
    type: File
    outputBinding:
      glob: "*.html"


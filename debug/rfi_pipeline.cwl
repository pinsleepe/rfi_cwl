cwlVersion: v1.0
class: CommandLineTool
label: Example trivial wrapper for python compiler
hints:
  DockerRequirement:
    dockerPull: rfipip
baseCommand: [/usr/bin/python, /home/rfi_cwl.py, --path]
inputs:
  src:
    type: File
    inputBinding:
      position: 1
outputs:
  classfile:
    type: File
    outputBinding:
      glob: "*.h5"


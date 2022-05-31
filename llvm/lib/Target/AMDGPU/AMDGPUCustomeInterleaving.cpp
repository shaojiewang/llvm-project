//===--- AMDGPUCustomInterleaving.cpp - AMDGPU Custom Interleaving  -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation for interleaving inside
///       a GEMM hot loop.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUCustomeInterleaving.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

using namespace llvm;

namespace {

class CustomInterleaving : public ScheduleDAGMutation {
public:
  CustomInterleaving() {}
  void apply(ScheduleDAGInstrs *DAG) override;
};
// Try recognize a CONV hot loop.
// The 0th SUnit would be an inline asm.
// The last SUnit would be an S_CBRANCH_SCC1.
bool identifyGEMMHotLoop(ScheduleDAGInstrs *DAG) {
  bool gotBegin = false;
  bool gotEnd = false;

  const SUnit &SU = DAG->SUnits[0];
  if (SU.isInstr()) {
    const MachineInstr *MI = SU.getInstr();
    if (MI->getOpcode() == AMDGPU::DS_READ2_B64_gfx9) {
      llvm::errs() << "find ds read\n";
      gotBegin = true;
    }
  }

  if (gotBegin) {
    if (DAG->ExitSU.getInstr() != nullptr) {
      const MachineInstr *MI = DAG->ExitSU.getInstr();
      if (MI->getOpcode() == AMDGPU::S_CBRANCH_SCC1) {
        gotEnd = true;
      }
    }
  }

  return (gotBegin && gotEnd);
}

void printNodeName(const SUnit &SU, const SUnit &EntrySU, const SUnit &ExitSU)
{
  if (&SU == &EntrySU)
    llvm::errs() << "EntrySU" << "\n";
  else if (&SU == &ExitSU)
    llvm::errs() << "ExitSU"<< "\n";
  else
    llvm::errs() << "SU(" << SU.NodeNum << ")\n";
}
#if 0
void dumpTRI(const TargetRegisterInfo *TRI){
  switch (getKind()) {
  case Data:   dbgs() << "Data"; break;
  case Anti:   dbgs() << "Anti"; break;
  case Output: dbgs() << "Out "; break;
  case Order:  dbgs() << "Ord "; break;
  }

  switch (getKind()) {
  case Data:
    dbgs() << " Latency=" << getLatency();
    if (TRI && isAssignedRegDep())
      dbgs() << " Reg=" << printReg(getReg(), TRI);
    break;
  case Anti:
  case Output:
    dbgs() << " Latency=" << getLatency();
    break;
  case Order:
    dbgs() << " Latency=" << getLatency();
    switch(Contents.OrdKind) {
    case Barrier:      dbgs() << " Barrier"; break;
    case MayAliasMem:
    case MustAliasMem: dbgs() << " Memory"; break;
    case Artificial:   dbgs() << " Artificial"; break;
    case Weak:         dbgs() << " Weak"; break;
    case Cluster:      dbgs() << " Cluster"; break;
    }
    break;
  }
}
#endif

bool checkInstType(const SUnit SU, int check_type)
{
  bool res = false;
  if (SU.isInstr()) {
    const MachineInstr *MI = SU.getInstr();
    if (MI->getOpcode() == check_type) {
      res = true;
    }
    if(MI->isInlineAsm())
    {

    }
  }
  return res;
}

void CustomInterleaving::apply(ScheduleDAGInstrs *DAG) {
  if (!identifyGEMMHotLoop(DAG))
    return;

  llvm::errs() << "Inside a GEMM hot loop DAG.\n";

  llvm::errs() << "Before adding cluster edges.\n";
  for (const SUnit &SU : DAG->SUnits) {
    //DAG->dumpNodeAll(SU);
    printNodeName(SU, DAG->EntrySU, DAG->ExitSU);
    llvm::errs() << DAG->getGraphNodeLabel(&SU);
    llvm::errs() << "==========\n";
  }

  int count_ds_read = 0;
  int count_ds_write = 0;
  int count_buffer_load = 0;
  int count_mfma = 0;
  for(const auto SU : DAG->SUnits)
  {
    if(checkInstType(SU, AMDGPU::DS_READ2_B64_gfx9))
    {
      count_ds_read++;
    }
    if(checkInstType(SU, AMDGPU::V_MFMA_F32_32X32X8F16_mac_e64))
    {
      count_mfma++;
    }
    if(checkInstType(SU, AMDGPU::DS_WRITE2_B64_gfx9))
    {
      count_ds_write++;
    }
    if(checkInstType(SU, AMDGPU::BUFFER_LOAD_DWORDX4_OFFEN))
    {
      count_buffer_load++;
    }
    if(checkInstType(SU, AMDGPU::INLINEASM))
    {
      //llvm::errs() << SU.getInstr()->getOperand(0).getSymbolName() << "\n";
      StringRef inline_str = SU.getInstr()->getOperand(0).getSymbolName();
      StringRef s_barrier_str = "s_barrier";
      size_t res_pos = inline_str.find(s_barrier_str);
      if(res_pos != StringRef::npos)
      {
        llvm::errs() << inline_str << "\n";
      }

    }
  }

  llvm::errs() << "count_ds_read:" << count_ds_read << ",count_mfma:" << count_mfma << "\n";

  //llvm::errs() << "Add some cluster edges.\n";
  //DAG->addEdge(&DAG->SUnits[5], SDep(&DAG->SUnits[3], SDep::Cluster));
  //DAG->addEdge(&DAG->SUnits[5], SDep(&DAG->SUnits[3], SDep::Artificial));
  //DAG->addEdge(&DAG->SUnits[6], SDep(&DAG->SUnits[3], SDep::Cluster));
  //DAG->addEdge(&DAG->SUnits[6], SDep(&DAG->SUnits[3], SDep::Artificial));

  //llvm::errs() << "After adding cluster edges.\n";
  //for (SUnit &SU : DAG->SUnits) {
  //  DAG->dumpNodeAll(SU);
  //  llvm::errs() << "==========\n";
  //}
}

} // end namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation>
createAMDGPUCustomInterleavingDAGMutation() {
  return std::make_unique<CustomInterleaving>();
}

} // end namespace llvm

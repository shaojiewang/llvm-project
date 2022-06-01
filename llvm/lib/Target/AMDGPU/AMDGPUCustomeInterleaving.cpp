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

#include <unordered_map>

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

static bool isDSRead(const SUnit &SU) {
  MachineInstr *MI = SU.getInstr();
  return (SIInstrInfo::isDS(*MI) && (MI->mayLoad()));
}
// Try recognize a CONV hot loop.
// The 0th SUnit would be an inline asm.
// The last SUnit would be an S_CBRANCH_SCC1.
bool identifyGEMMHotLoop(ScheduleDAGInstrs *DAG) {
  bool gotBegin = false;
  bool gotEnd = false;

  const SUnit &SU = DAG->SUnits[0];
  if (SU.isInstr()) {
    const MachineInstr *MI = SU.getInstr();
    if (isDSRead(SU)) {
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


static bool isDSWrite(const SUnit &SU) {
  MachineInstr *MI = SU.getInstr();
  return (SIInstrInfo::isDS(*MI) && (MI->mayStore()));
}

static bool isMFMA(const SUnit &SU) {
  return SIInstrInfo::isMAI(*SU.getInstr());
}

static bool isVMEMLoad(const SUnit &SU) {
  MachineInstr *MI = SU.getInstr();
  return (SIInstrInfo::isVMEM(*MI) && (MI->mayLoad()));
}

static bool isVMEMStore(const SUnit &SU) {
  MachineInstr *MI = SU.getInstr();
  return (SIInstrInfo::isVMEM(*MI) && (MI->mayStore()));
}

static bool isVMUL(const SUnit &SU) {
  const MachineInstr *MI = SU.getInstr();
  if (MI->getOpcode() == AMDGPU::V_MUL_LO_I32_e64 
      || MI->getOpcode() == AMDGPU::V_MUL_HI_I32_e64
      || MI->getOpcode() == AMDGPU::V_MUL_LO_U32_e64 
      || MI->getOpcode() == AMDGPU::V_MUL_HI_U32_e64) {
    return true;
  }
  else{
    return false;
  }
}

static bool isSBarrier(const SUnit &SU) {
  StringRef inline_str = SU.getInstr()->getOperand(0).getSymbolName();
  StringRef s_barrier_str = "s_barrier";
  size_t res_pos = inline_str.find(s_barrier_str);
  if(res_pos != StringRef::npos)
  {
    //llvm::errs() << inline_str << "\n";
    return true;
  }
  else
  {
    return false;
  }
}

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

  int DSReadCount = 0;
  int DSWriteCount = 0;
  int VMEMLoadCount = 0;
  int MFMACount = 0;
  int VMULCount = 0;
  int SBarrierCount = 0;
  int VMEMStoreCount = 0;
  int OthersCount = 0;

  SmallVector<SUnit*, 16> DSReads;
  SmallVector<SUnit*, 8> DSWrites;
  SmallVector<SUnit*, 8> VMEMLoads;
  SmallVector<SUnit*, 8> VMEMStores;
  SmallVector<SUnit*, 30> VMULValus;
  SmallVector<SUnit*, 8> SBarriers;
  SmallVector<SUnit*, 100> Others;
  SmallVector<SUnit*, 64> MFMAs;
  SmallVector<SUnit*, 200> InstructionToInterLeave;

  std::unordered_map<SUnit*, int> InstLatMap;

  for(SUnit &SU : DAG->SUnits)
  {
    if (isDSRead(SU)) {
      DSReadCount++;
      DSReads.push_back(&SU);
      InstructionToInterLeave.push_back(&SU);
      InstLatMap.insert({&SU, 4});
    } else if (isDSWrite(SU)) {
      DSWriteCount++;
      DSWrites.push_back(&SU);
      InstructionToInterLeave.push_back(&SU);
      InstLatMap.insert({&SU, 30});
    } else if (isMFMA(SU)) {
      MFMACount++;
      MFMAs.push_back(&SU);
    } else if (isVMEMLoad(SU)) {
      VMEMLoadCount++;
      VMEMLoads.push_back(&SU);
      InstructionToInterLeave.push_back(&SU);
      InstLatMap.insert({&SU, 30});
    } else if (isVMEMStore(SU)) {
      VMEMStoreCount++;
      VMEMStores.push_back(&SU);
      InstructionToInterLeave.push_back(&SU);
      InstLatMap.insert({&SU, 30});
    } else if (isVMUL(SU)) {
      VMULCount++;
      VMULValus.push_back(&SU);
      InstructionToInterLeave.push_back(&SU);
      InstLatMap.insert({&SU, 8});
    } else if (isSBarrier(SU)) {
      SBarrierCount++;
      SBarriers.push_back(&SU);
      InstructionToInterLeave.push_back(&SU);
      InstLatMap.insert({&SU, 55});
    } else {
      OthersCount++;
      Others.push_back(&SU);
      InstructionToInterLeave.push_back(&SU);
      InstLatMap.insert({&SU, 4});
    }
  }

  llvm::errs() << "DSRead instruction count: " << DSReadCount << "\n";
  llvm::errs() << "DSWrite instruction count: " << DSWriteCount << "\n";
  llvm::errs() << "VMEMLoad instruction count: " << VMEMLoadCount << "\n";
  llvm::errs() << "VMEMStore instruction count: " << VMEMStoreCount << "\n";
  llvm::errs() << "MFMA instruction count: " << MFMACount << "\n";
  llvm::errs() << "SBarrier instruction count: " << SBarrierCount << "\n";
  llvm::errs() << "VMUL instruction count: " << VMULCount << "\n";
  llvm::errs() << "Other instruction count: " << OthersCount << "\n";

  //assert(VMEMStoreCount == 0);
  assert(MFMACount * 56 > (VMEMLoadCount * 30 + DSWriteCount * 30 + DSReadCount * 4));

  int64_t MFMAIter = MFMAs.size() - 1;

  int MFMALatShadow = 56;
  int InstToInterleaveIter = InstructionToInterLeave.size() - 1;

#if 0
  for(int i_mfma = MFMAIter - 1; i_mfma > 0; i_mfma--)
  {
    MFMALatShadow = 56;
    SUnit* MFMASU = MFMAs[i_mfma];
    while(MFMALatShadow > 0 && InstToInterleaveIter > 0)
    {
      SUnit* InstBetweenMFMA = InstructionToInterLeave[InstToInterleaveIter--];
      MFMALatShadow -= InstLatMap[InstBetweenMFMA];
      printNodeName(*InstBetweenMFMA, DAG->EntrySU, DAG->ExitSU);
      llvm::errs() << DAG->getGraphNodeLabel(InstBetweenMFMA) << "\n";
      llvm::errs() << MFMALatShadow << "," << InstToInterleaveIter << "\n";
    }
    InstToInterleaveIter++;
    SUnit* InstMutation = InstructionToInterLeave[InstToInterleaveIter];
    DAG->addEdge(MFMASU, SDep(InstMutation, SDep::Artificial));
  }

#else

  assert(VMEMStoreCount == 0);

  // Determine the order of interleaving.
  int64_t DSReadPriority, DSWritePriority, VMEMLoadPriority;
  DSReadPriority = DSWritePriority = VMEMLoadPriority = -1;
  auto NotAssignedPriority = [](int64_t prio) { return prio < 0; };

  int64_t CurrentPriority, TotalPriority;
  CurrentPriority = TotalPriority = 0;

  // Starting backward.
  int64_t SUIter = DAG->SUnits.size() - 1;
  while (SUIter >= 0) {
    SUnit &SU = DAG->SUnits[SUIter--];
    if (isDSRead(SU) && NotAssignedPriority(DSReadPriority)) {
      DSReadPriority = CurrentPriority++;
    } else if (isDSWrite(SU) && NotAssignedPriority(DSWritePriority)) {
      DSWritePriority = CurrentPriority++;
    } else if (isVMEMLoad(SU) && NotAssignedPriority(VMEMLoadPriority)) {
      VMEMLoadPriority = CurrentPriority++;
    }
  }
  TotalPriority = CurrentPriority;

#if 1
  llvm::errs() << "DSReadPriority: " << DSReadPriority << "\n";
  llvm::errs() << "DSWritePriority: " << DSWritePriority << "\n";
  llvm::errs() << "VMEMLoadPriority: " << VMEMLoadPriority << "\n";
#endif

#if 0
  llvm::errs() << "Add some artificial edges.\n";
#endif

  int64_t MFMAIter = MFMAs.size() - 1;

  // Reset CurrentPriority.
  CurrentPriority = 0;

  // Iterate through all different instruction groups to be interleaved with MFMA.
  while (CurrentPriority < TotalPriority) {
    if (CurrentPriority == VMEMLoadPriority) {
      // Interleave MFMA with buffer_loads.
      int64_t VMEMLoadIter = VMEMLoads.size() - 1;
      while ((VMEMLoadIter >= 0) && (MFMAIter >= 0)) {
        SUnit* VMEMLoadSU = VMEMLoads[VMEMLoadIter--];
        SUnit* MFMASU = MFMAs[MFMAIter--];
        DAG->addEdge(MFMASU, SDep(VMEMLoadSU, SDep::Artificial));
      }
    } else if (CurrentPriority == DSWritePriority) {
      // Interleave MFMA with ds_writes.
      int64_t DSWriteIter = DSWrites.size() - 1;
      while ((DSWriteIter >= 0) && (MFMAIter >= 0)) {
        SUnit* DSWriteSU = DSWrites[DSWriteIter--];
        SUnit* MFMASU = MFMAs[MFMAIter--];
        DAG->addEdge(MFMASU, SDep(DSWriteSU, SDep::Artificial));
      }
    } else if (CurrentPriority == DSReadPriority) {
      // Interleave MFMA with ds_reads.
      int64_t DSReadIter = DSReads.size() - 1;
      while ((DSReadIter >= 0) && (MFMAIter >= 0)) {
        SUnit* DSReadSU = DSReads[DSReadIter--];
        SUnit* MFMASU = MFMAs[MFMAIter--];
        DAG->addEdge(MFMASU, SDep(DSReadSU, SDep::Artificial));
      }
    }

    // Move to the next instruction groups.
    ++CurrentPriority;
  }
#endif
  // llvm::errs() << "After adding cluster edges.\n";
  // for (const SUnit &SU : DAG->SUnits) {
  //   //DAG->dumpNodeAll(SU);
  //   printNodeName(SU, DAG->EntrySU, DAG->ExitSU);
  //   llvm::errs() << DAG->getGraphNodeLabel(&SU);
  //   llvm::errs() << "==========\n";
  // }
}

} // end namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation>
createAMDGPUCustomInterleavingDAGMutation() {
  return std::make_unique<CustomInterleaving>();
}

} // end namespace llvm

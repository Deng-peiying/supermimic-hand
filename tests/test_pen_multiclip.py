"""多片段边界：未来目标、局部相位、采样和想象末端不能串段。"""
from types import SimpleNamespace
import numpy as np
import torch

from supermimic.tasks.pen_spin.data.sampling import MultiClipSampler
from supermimic.tasks.pen_spin.policy.reference import future_for_env, reference_phase, reference_end


def test_reference_never_crosses_clip_boundary():
    env = SimpleNamespace(ref=torch.arange(12.)[:,None].expand(-1,57),length=11,stage='full',
                          reference_starts=torch.tensor([0]*5+[5]*7),
                          reference_ends=torch.tensor([4]*5+[11]*7))
    frame=torch.tensor([3,4,5,10])
    future=future_for_env(env,frame)
    assert torch.all(future[:2]==4)
    assert future[2,0,0]==6 and future[2,-1,0]==11
    torch.testing.assert_close(reference_phase(env,frame).flatten(),torch.tensor([.75,1.,0.,5/6]))
    nxt=torch.minimum(frame+1,reference_end(env,frame))
    assert nxt.tolist()==[4,4,6,11]


def test_multiclip_sampler_excludes_terminal_and_balances_clips():
    torch.manual_seed(42)
    clip=SimpleNamespace(starts=np.array([0,5]),ends=np.array([4,205]),q=np.zeros((206,22)))
    for mode in ('start','uniform','return'):
        sampler=MultiClipSampler(3,clip,'cpu',.99,mode,.124)
        frames=sampler.sample(4000)
        assert not ((frames==4)|(frames==205)).any()
        assert 1800<int((frames<5).sum())<2200
        if mode=='start': assert set(frames.tolist())=={0,5}
    result=sampler.record(torch.ones(3),torch.tensor([0,5,204]),torch.ones(3,dtype=torch.bool))
    assert result['episodes']==3 and result['episode_steps']==1
    assert torch.isfinite(sampler.value).all()

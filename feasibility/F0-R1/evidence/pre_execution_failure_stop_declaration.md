\# F0-R1 Pre-Execution Failure and Stop Declaration



Amendment: F0-R1-A1



This declaration is frozen before the governed R1 execution.



\## Mandatory stop and preservation rules



1\. The sealed final test artifact `untouched\_final\_test\_R1.npz` must not be accessed, inspected, transmitted, or used by R1.2-R1.5.

2\. Seeds must not be changed after execution results are observed.

3\. Frozen decision thresholds and protocol parameters must not be changed after execution results are observed.

4\. Pinned dependencies and the accepted environment lock must not be changed to force a passing result.

5\. If a governed test fails, the first failure and its terminal log must be preserved.

6\. Failed runs must not be deleted or overwritten in order to obtain a successful rerun.

7\. Any command requiring access to the sealed final test is a stop condition.

8\. Any ad-hoc latest-package installation or unapproved dependency change is a stop condition.

9\. If the required CPU smoke test, deterministic rerun, Flower simulation, or CVXPY optimization smoke test fails, governed execution stops and the failure is preserved.

10\. Optimizer failure during the R1 procedure must be recorded and handled using the specified local-reference fallback; it must not be reported as optimizer success.

11\. Synthetic smoke results are integration evidence only and must not be presented as research evidence.

12\. R1.6 GREEN/CONDITIONAL/RED is a technical recommendation only. It does not authorize Phase 1; supervisor approval is required.

13\. The native Ubuntu requirement in the governing F0 procedure remains a prerequisite for the governed research run. This declaration does not override that requirement.



\## Execution integrity rule



If any mandatory stop condition is encountered, execution stops at that point and the failure/evidence is preserved. No rerun is permitted merely to replace the failed outcome.



Declaration status: PRE-EXECUTION / FROZEN
